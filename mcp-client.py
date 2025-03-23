import os
import json
import time
import uuid
import asyncio
import subprocess
import threading
import logging
import signal
import atexit
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Header, Body, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Import the MCP SDK
try:
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp import StdioServerParameters
except ImportError:
    logger = logging.getLogger('mcp_client')
    logger.warning("MCP SDK not found, installing...")
    subprocess.run(["pip", "install", "mcp"], check=True)
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp import StdioServerParameters

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('mcp_client')

# Load environment variables
load_dotenv()

# Load server configurations from JSON file
def load_server_configs():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "servers.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading server configurations: {str(e)}")
            return {"mcpServers": {}}
    else:
        logger.warning(f"Server configuration file not found at {config_path}")
        return {"mcpServers": {}}

# Initialize FastAPI
app = FastAPI(
    title="MCP Client Service",
    description="A service that acts as a bridge between n8n and MCP servers",
    version="1.0.0",
)

# CORS middleware configured with environment variables
# Default to common development origins, but allow configuration for production
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,http://localhost:8080")
origins = allowed_origins.split(",")

logger.info(f"Configuring CORS with allowed origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication
X_API_KEY = os.getenv("X_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != X_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
        )
    return api_key

# MCP Models
class MCPToolCall(BaseModel):
    tool_name: str
    tool_args: Dict[str, Any]
    tool_call_id: Optional[str] = None

class MCPServerDefinition(BaseModel):
    package: str
    description: Optional[str] = "MCP Server"
    env: Optional[Dict[str, str]] = None
    args: Optional[List[str]] = None

# Global state
mcp_processes = {}  # Store running MCP server processes
mcp_sessions = {}   # Store active MCP client sessions
used_ports = set()  # Track used ports for MCP servers
server_configs = load_server_configs()  # Load server configurations

# Find a free port to use
def find_free_port(start_port, end_port):
    import socket
    
    for port in range(start_port, end_port + 1):
        if port in used_ports:
            continue
            
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except socket.error:
                pass
    
    return None

# Get the next available port
def get_next_available_port():
    port = 3000
    while port in used_ports:
        port += 1
    used_ports.add(port)
    return port

# Start an MCP server as an NPX process
def start_mcp_server(server_id: str, server_config: Dict[str, Any]):
    """
    Start an MCP server process.
    Returns process information if successful, None otherwise.
    """
    try:
        # Create logs directory if it doesn't exist
        if not os.path.exists("logs"):
            os.makedirs("logs")
        
        # Prepare command
        cmd = [server_config["command"]]
        cmd.extend(server_config.get("args", []))
        
        # Let the server use its default transport mode (stdio)
        # We'll handle both stdio and HTTP with the appropriate client
        
        # Add port flag to specify a unique port (for HTTP mode if used)
        port = get_next_available_port()
        
        # Set up environment variables
        env = os.environ.copy()
        for key, value in server_config.get("env", {}).items():
            # Handle environment variable references
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var_name = value[2:-1]
                env_var_value = os.getenv(env_var_name)
                if env_var_value:
                    logger.info(f"Set environment variable {key} for server {server_id}")
                    env[key] = env_var_value
                else:
                    logger.warning(f"Environment variable {env_var_name} not found for {key}")
            else:
                env[key] = value
        
        # Open log files
        stdout_log = open(f"logs/{server_id}_stdout.log", "w")
        stderr_log = open(f"logs/{server_id}_stderr.log", "w")
        
        # Start the process
        logger.info(f"Starting MCP server {server_id} with command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,  # Use PIPE to keep stdout accessible for stdio communication
            stderr=stderr_log,
            stdin=subprocess.PIPE,   # Use PIPE to keep stdin accessible for stdio communication
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Create a thread to log stdout without consuming it
        def log_stdout():
            """Log stdout to file without consuming the pipe"""
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                stdout_log.write(line)
                stdout_log.flush()
        
        # Start the logging thread
        stdout_thread = threading.Thread(target=log_stdout, daemon=True)
        stdout_thread.start()
        
        # Store process information
        mcp_processes[server_id] = {
            "process": process,
            "running": True,
            "port": port,
            "stdout_log": stdout_log,
            "stderr_log": stderr_log,
            "stdout_thread": stdout_thread,
            "started_at": time.time()
        }
        
        # Wait for server to start
        logger.info(f"Started MCP server {server_id} (PID: {process.pid}) on port {port}")
        time.sleep(3)  # Give the server time to start
        
        return mcp_processes[server_id]
    except Exception as e:
        logger.error(f"Failed to start MCP server {server_id}: {str(e)}")
        return None

# Stop an MCP server process
def stop_mcp_server(server_id: str):
    if server_id not in mcp_processes:
        logger.warning(f"MCP server {server_id} not found")
        return False
    
    process_info = mcp_processes[server_id]
    process = process_info.get("process")
    
    if not process or process.poll() is not None:
        logger.warning(f"MCP server {server_id} is not running")
        return False
    
    try:
        # Terminate the process
        logger.info(f"Stopping MCP server {server_id}")
        process.terminate()
        
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"Server {server_id} did not terminate, killing")
            process.kill()
            process.wait()
        
        # Remove from sessions cache if present
        if server_id in mcp_sessions:
            del mcp_sessions[server_id]
        
        # Close log files
        if "stdout_log" in process_info and process_info["stdout_log"]:
            try:
                process_info["stdout_log"].close()
            except Exception as e:
                logger.warning(f"Error closing stdout file for server {server_id}: {str(e)}")
        
        if "stderr_log" in process_info and process_info["stderr_log"]:
            try:
                process_info["stderr_log"].close()
            except Exception as e:
                logger.warning(f"Error closing stderr file for server {server_id}: {str(e)}")
        
        # Update the process info
        process_info["running"] = False
        
        # Free the port
        if "port" in process_info and process_info["port"] in used_ports:
            used_ports.remove(process_info["port"])
        
        return True
    except Exception as e:
        logger.error(f"Error stopping server {server_id}: {str(e)}")
        return False

# Clean up processes on shutdown
def cleanup_processes():
    logger.info("Cleaning up MCP server processes...")
    for server_id, info in list(mcp_processes.items()):
        process = info.get("process")
        if process and info.get("running", False):
            logger.info(f"Terminating MCP server {server_id} (PID: {process.pid})")
            try:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"MCP server {server_id} did not terminate, killing...")
                    process.kill()
                
                # Close log files
                if "stdout_log" in info and info["stdout_log"]:
                    try:
                        info["stdout_log"].close()
                        logger.debug(f"Closed stdout file for server {server_id}")
                    except:
                        pass
                
                if "stderr_log" in info and info["stderr_log"]:
                    try:
                        info["stderr_log"].close()
                        logger.debug(f"Closed stderr file for server {server_id}")
                    except:
                        pass
                
            except Exception as e:
                logger.error(f"Error cleaning up server {server_id}: {str(e)}")
            
            info["running"] = False

atexit.register(cleanup_processes)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, lambda sig, frame: (cleanup_processes(), exit(0)))
signal.signal(signal.SIGTERM, lambda sig, frame: (cleanup_processes(), exit(0)))

# Start all configured servers on startup
def start_configured_servers():
    for server_id, config in server_configs.get("mcpServers", {}).items():
        logger.info(f"Starting configured MCP server: {server_id}")
        start_mcp_server(server_id, config)

# MCP Client Session Management
async def get_mcp_client_session(server_id: str):
    """
    Get or create an MCP client session for a server.
    Returns a session if successful, None otherwise.
    """
    try:
        # Check if we already have a valid session
        if server_id in mcp_sessions and mcp_sessions[server_id].get("valid", False):
            logger.info(f"Using existing MCP client session for server {server_id}")
            return mcp_sessions[server_id]["session"]
        
        # Check if the server is running
        if server_id not in mcp_processes or not mcp_processes[server_id].get("running", False):
            # Try to start the server
            logger.info(f"Server {server_id} not running, attempting to start")
            if server_id not in server_configs["mcpServers"]:
                logger.error(f"Server {server_id} not found in configuration")
                return None
            
            server_config = server_configs["mcpServers"][server_id]
            process_info = start_mcp_server(server_id, server_config)
            
            if not process_info:
                logger.error(f"Failed to start server {server_id}")
                return None
        
        # Instead of trying to communicate with the existing process,
        # let's start a fresh process using the MCP SDK's stdio_client
        try:
            logger.info(f"Creating new MCP client session for server {server_id}")
            
            # Get the server configuration
            server_config = server_configs["mcpServers"][server_id]
            cmd = server_config.get("command", "").split()
            args = server_config.get("args", [])
            
            # Prepare environment variables
            env = {}
            for key, value in server_config.get("env", {}).items():
                # Handle environment variable references
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var_name = value[2:-1]
                    env_var_value = os.getenv(env_var_name)
                    if env_var_value:
                        env[key] = env_var_value
                    else:
                        logger.warning(f"Environment variable {env_var_name} not found for {key}")
                else:
                    env[key] = value
            
            # Create server parameters for a new process
            server_params = StdioServerParameters(
                command=cmd[0],
                args=args,
                env=env
            )
            
            # Add the Node.js executable path to the environment variables for the server process
            node_bin_path = os.path.dirname(cmd[0])  # Get directory containing npx
            if 'PATH' in os.environ:
                server_params.env['PATH'] = f"{node_bin_path}:{os.environ['PATH']}"
            else:
                server_params.env['PATH'] = node_bin_path
            
            logger.info(f"Set PATH for {server_id} to include Node.js bin directory: {node_bin_path}")
            
            # Use the stdio_client as a context manager to create a new process
            # and properly handle the communication
            logger.info(f"Starting new process for {server_id} with stdio_client")
            
            # Create a new process and session
            async def create_and_initialize_session():
                logger.info(f"Starting stdio_client for {server_id}")
                async with stdio_client(server_params) as (read, write):
                    logger.info(f"stdio_client started for {server_id}, creating session")
                    session = ClientSession(read, write)
                    logger.info(f"Session created for {server_id}, initializing...")
                    await session.initialize()
                    logger.info(f"Session initialized for {server_id}")
                    return session
            
            # Set a timeout for initialization
            try:
                logger.info(f"Setting timeout for {server_id} session initialization")
                async with asyncio.timeout(15.0):  # 15 second timeout for initialization
                    session = await create_and_initialize_session()
                logger.info(f"Session initialization completed for {server_id}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout initializing session for {server_id}")
                raise Exception(f"Timeout initializing session for {server_id}")
            
            # Store the session
            mcp_sessions[server_id] = {
                "session": session,
                "valid": True,
                "timestamp": time.time()
            }
            
            return session
        except Exception as e:
            logger.error(f"Failed to create stdio session for {server_id}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    except Exception as e:
        import traceback
        logger.error(f"Failed to create MCP client session: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

# API endpoints
@app.get("/servers", response_model=Dict[str, Any])
async def list_servers(api_key: str = Depends(verify_api_key)):
    """
    List all configured MCP servers.
    This endpoint returns information about all servers defined in the servers.json file.
    """
    server_list = []
    
    for server_id, config in server_configs.get("mcpServers", {}).items():
        # Check if the server is running
        status = "running" if (
            server_id in mcp_processes and 
            mcp_processes[server_id].get("process") and 
            mcp_processes[server_id].get("process").poll() is None
        ) else "not_running"
        
        # Extract only the necessary information for the AI agent
        server_info = {
            "id": server_id,
            "name": config.get("name", server_id),
            "description": config.get("description", "MCP Server"),
            "status": status,
            "package": config.get("package", "")
        }
        server_list.append(server_info)
    
    # Sort servers by ID
    server_list.sort(key=lambda x: x["id"])
    
    return {
        "servers": server_list,
        "mcp_sdk_version": "unknown"
    }

@app.get("/servers/{server_id}/tools", tags=["MCP Server Tools"])
async def list_server_tools(server_id: str, api_key: str = Depends(verify_api_key)):
    """
    List available tools for an MCP server.
    """
    try:
        # Check if the server exists in configuration
        if server_id not in server_configs["mcpServers"]:
            logger.error(f"Server {server_id} not found in configuration")
            raise HTTPException(
                status_code=404, 
                detail=f"Server {server_id} not found"
            )
        
        # Get the server configuration
        server_config = server_configs["mcpServers"][server_id]
        
        # Extract command and args
        cmd = server_config.get("command", "npx")
        args = server_config.get("args", [])
        
        # Prepare environment variables
        env = {}
        for key, value in server_config.get("env", {}).items():
            # Handle environment variable references
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var_name = value[2:-1]
                env_var_value = os.getenv(env_var_name)
                if env_var_value:
                    env[key] = env_var_value
                else:
                    logger.warning(f"Environment variable {env_var_name} not found for server {server_id}")
            else:
                env[key] = value
        
        # Create server parameters
        server_params = StdioServerParameters(
            command=cmd,
            args=args,
            env=env
        )
        
        # Add the Node.js executable path to the environment variables for the server process
        node_bin_path = os.path.dirname(cmd)  # Get directory containing npx
        if 'PATH' in os.environ:
            server_params.env['PATH'] = f"{node_bin_path}:{os.environ['PATH']}"
        else:
            server_params.env['PATH'] = node_bin_path
        
        logger.info(f"Set PATH for {server_id} to include Node.js bin directory: {node_bin_path}")
        
        # Set a timeout for the entire operation
        try:
            async with asyncio.timeout(15.0):  # 15 second timeout
                # Follow the example pattern exactly
                logger.info(f"Creating new process for {server_id} with stdio_client")
                async with stdio_client(server_params) as (read, write):
                    logger.info(f"Created stdio client for {server_id}, initializing session")
                    async with ClientSession(read, write) as session:
                        # Initialize the session
                        logger.info(f"Initializing session for {server_id}")
                        await session.initialize()
                        
                        # List tools with the session
                        logger.info(f"Listing tools for server {server_id}")
                        tools_response = await session.list_tools()
                        
                        # Return the tools
                        tools = []
                        for tool in tools_response.tools:
                            tool_info = {
                                "name": tool.name,
                                "description": tool.description
                            }
                            
                            # Handle input schema based on its type
                            schema_found = False
                            
                            # Check for input_schema attribute (snake_case)
                            if hasattr(tool, "input_schema") and tool.input_schema:
                                if hasattr(tool.input_schema, "model_json_schema"):
                                    tool_info["inputSchema"] = tool.input_schema.model_json_schema()
                                    schema_found = True
                                else:
                                    tool_info["inputSchema"] = tool.input_schema
                                    schema_found = True
                            
                            # Check for inputSchema attribute (camelCase) - used by Brave search server
                            elif hasattr(tool, "inputSchema") and tool.inputSchema:
                                tool_info["inputSchema"] = tool.inputSchema
                                schema_found = True
                            
                            # If no schema is available, provide a standardized empty schema
                            # This allows n8n to at least show the tool without breaking
                            if not schema_found:
                                logger.warning(f"Tool {tool.name} from server {server_id} does not provide an input schema")
                                tool_info["inputSchema"] = {
                                    "type": "object",
                                    "properties": {},
                                    "description": f"Schema not provided by the MCP server. Please refer to documentation for {tool.name}."
                                }
                            
                            tools.append(tool_info)
                        
                        return {"tools": tools}
        except asyncio.TimeoutError:
            logger.error(f"Timeout while listing tools for server {server_id}")
            raise HTTPException(
                status_code=504, 
                detail="Timeout while communicating with the MCP server"
            )
    except Exception as e:
        logger.error(f"Error listing tools for server {server_id}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error listing tools: {str(e)}"
        )

@app.post("/servers/{server_id}/tools/{tool_name}", tags=["MCP Server Tools"])
async def call_server_tool(
    server_id: str, 
    tool_name: str, 
    tool_args: Dict[str, Any], 
    api_key: str = Depends(verify_api_key)
):
    """
    Call a tool on an MCP server with the provided arguments.
    """
    try:
        # Check if the server exists in configuration
        if server_id not in server_configs["mcpServers"]:
            logger.error(f"Server {server_id} not found in configuration")
            raise HTTPException(
                status_code=404, 
                detail=f"Server {server_id} not found"
            )
        
        # Get the server configuration
        server_config = server_configs["mcpServers"][server_id]
        
        # Extract command and args
        cmd = server_config.get("command", "npx")
        args = server_config.get("args", [])
        
        # Prepare environment variables
        env = {}
        for key, value in server_config.get("env", {}).items():
            # Handle environment variable references
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var_name = value[2:-1]
                env_var_value = os.getenv(env_var_name)
                if env_var_value:
                    env[key] = env_var_value
                else:
                    logger.warning(f"Environment variable {env_var_name} not found for server {server_id}")
            else:
                env[key] = value
        
        # Create server parameters
        server_params = StdioServerParameters(
            command=cmd,
            args=args,
            env=env
        )
        
        # Add the Node.js executable path to the environment variables for the server process
        node_bin_path = os.path.dirname(cmd)  # Get directory containing npx
        if 'PATH' in os.environ:
            server_params.env['PATH'] = f"{node_bin_path}:{os.environ['PATH']}"
        else:
            server_params.env['PATH'] = node_bin_path
        
        logger.info(f"Set PATH for {server_id} to include Node.js bin directory: {node_bin_path}")
        
        # Set a timeout for the entire operation
        try:
            async with asyncio.timeout(30.0):  # 30 second timeout for tool calls
                # Follow the example pattern exactly
                logger.info(f"Creating new process for {server_id} to call tool {tool_name}")
                async with stdio_client(server_params) as (read, write):
                    logger.info(f"Created stdio client for {server_id}, initializing session")
                    async with ClientSession(read, write) as session:
                        # Initialize the session
                        logger.info(f"Initializing session for {server_id}")
                        await session.initialize()
                        
                        # Call the tool with the session
                        logger.info(f"Calling tool {tool_name} on server {server_id} with args: {tool_args}")
                        tool_response = await session.call_tool(tool_name, tool_args)
                        
                        # Process the response
                        response_content = []
                        for content_item in tool_response.content:
                            if hasattr(content_item, "text") and content_item.text:
                                response_content.append({"type": "text", "text": content_item.text})
                            elif hasattr(content_item, "json") and content_item.json:
                                response_content.append({"type": "json", "json": content_item.json})
                        
                        return {
                            "tool": tool_name,
                            "content": response_content
                        }
        except asyncio.TimeoutError:
            logger.error(f"Timeout while calling tool {tool_name} on server {server_id}")
            raise HTTPException(
                status_code=504, 
                detail="Timeout while communicating with the MCP server"
            )
    except Exception as e:
        logger.error(f"Error calling tool {tool_name} on server {server_id}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error calling tool: {str(e)}"
        )

@app.post("/servers/{server_id}/tool-call", response_model=Dict[str, Any])
async def call_tool(
    server_id: str,
    tool_call: MCPToolCall,
    api_key: str = Depends(verify_api_key)
):
    """
    Call a tool on an MCP server.
    Uses the MCP protocol to communicate with the server.
    """
    try:
        session = await get_mcp_client_session(server_id)
        
        # Generate a tool call ID if not provided
        call_id = tool_call.tool_call_id or str(uuid.uuid4())
        
        logger.info(f"Calling tool {tool_call.tool_name} on MCP server {server_id}")
        result = await session.invoke_tool(
            tool_call.tool_name,
            tool_call.tool_args
        )
        
        return {
            "tool_call_id": call_id,
            "result": result,
            "server_id": server_id,
            "tool_name": tool_call.tool_name
        }
    except Exception as e:
        logger.error(f"Error calling tool on MCP server {server_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error calling tool: {str(e)}"
        )

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Check the health of the service.
    """
    configured_count = len(server_configs.get("mcpServers", {}))
    
    # Count running servers
    running_count = 0
    for server_id, process_info in mcp_processes.items():
        if process_info.get("process") and process_info.get("process").poll() is None:
            running_count += 1
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "mcp_sdk_version": "unknown",
        "servers": {
            "configured": configured_count,
            "running": running_count
        }
    }

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # Start all configured servers
    start_configured_servers()
    
    port = int(os.getenv("PORT", "8888"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting MCP Client Service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)