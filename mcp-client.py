import os
import json
import logging
import subprocess
import time
import signal
import atexit
import uuid
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Header, Body, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import threading
from contextlib import asynccontextmanager

# Import the MCP SDK
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
import mcp.types as mcp_types

# Try to get the version - handle case where it's not available
try:
    from mcp import __version__ as mcp_version
except ImportError:
    mcp_version = "unknown"

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcp_client")

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

# Store running MCP server processes and connection info
mcp_processes: Dict[str, Dict[str, Any]] = {}

# Store MCP client session cache
mcp_sessions: Dict[str, Any] = {}

# Track used ports to avoid conflicts
used_ports = set()

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

# Start an MCP server as an NPX process
def start_mcp_server(server_id: str, server_config: Dict[str, Any]):
    if server_id in mcp_processes and mcp_processes[server_id].get("process") and \
       mcp_processes[server_id].get("process").poll() is None:
        logger.info(f"MCP server {server_id} is already running")
        return mcp_processes[server_id]
    
    # Find a free port for the server
    port = find_free_port(3000, 4000)
    if not port:
        logger.error("Failed to find a free port")
        return None
    
    used_ports.add(port)
    
    # Get the package name and command
    package = server_config.get("package")
    command = server_config.get("command", "npx")
    args = server_config.get("args", [])
    
    if not package and not (command and args):
        logger.error(f"Invalid server configuration for {server_id}: missing package or command")
        return None
    
    # Prepare the command
    if command == "npx":
        cmd = ["npx", "-y"]
        if package:
            cmd.append(package)
        elif args:
            cmd.extend(args[1:]) # Skip the '-y' that's already in cmd
    else:
        cmd = [command]
        if args:
            cmd.extend(args)
    
    # Prepare environment variables
    env_dict = dict(os.environ)
    env_dict["PORT"] = str(port)
    
    # Handle environment variable templates in the config
    if "env" in server_config:
        for env_name, env_value in server_config["env"].items():
            if isinstance(env_value, str) and env_value.startswith("${") and env_value.endswith("}"):
                # It's a template, get the value from environment
                env_var_name = env_value[2:-1]
                actual_value = os.environ.get(env_var_name)
                if actual_value:
                    env_dict[env_name] = actual_value
                else:
                    logger.warning(f"Environment variable {env_var_name} not found for {env_name}")
            else:
                env_dict[env_name] = env_value
    
    # Start the process
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env_dict
        )
        
        logger.info(f"Started MCP server {server_id} (PID: {process.pid}) on port {port}")
        
        # Store process info
        process_info = {
            "process": process,
            "port": port,
            "pid": process.pid,
            "started_at": time.time(),
            "running": True,
            "url": f"http://localhost:{port}/v1",
            "config": server_config
        }
        
        mcp_processes[server_id] = process_info
        
        # Give the server a moment to start
        time.sleep(1)
        
        return process_info
    except Exception as e:
        logger.error(f"Error starting MCP server {server_id}: {str(e)}")
        if port in used_ports:
            used_ports.remove(port)
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
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"MCP server {server_id} did not terminate, killing...")
                process.kill()
            info["running"] = False

atexit.register(cleanup_processes)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, lambda sig, frame: (cleanup_processes(), exit(0)))
signal.signal(signal.SIGTERM, lambda sig, frame: (cleanup_processes(), exit(0)))

# Start all configured servers on startup
def start_configured_servers():
    server_configs = load_server_configs()
    for server_id, config in server_configs.get("mcpServers", {}).items():
        logger.info(f"Starting configured MCP server: {server_id}")
        start_mcp_server(server_id, config)

# MCP Client Session Management
async def get_mcp_client_session(server_id: str):
    """
    Get or create an MCP client session for a server.
    Uses the SSE transport to communicate with the MCP server.
    """
    # Check if we already have a cached session
    if server_id in mcp_sessions and mcp_sessions[server_id].get("valid", False):
        return mcp_sessions[server_id]["session"]
    
    # Check if server exists in our configuration
    server_configs = load_server_configs()
    if "mcpServers" not in server_configs or server_id not in server_configs["mcpServers"]:
        raise HTTPException(
            status_code=404,
            detail=f"Server configuration for {server_id} not found"
        )
    
    # Start the server if it's not already running
    if server_id not in mcp_processes or not mcp_processes[server_id].get("running", False):
        server_config = server_configs["mcpServers"][server_id]
        process_info = start_mcp_server(server_id, server_config)
        if not process_info:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start server {server_id}"
            )
    
    # Get server URL
    process_info = mcp_processes[server_id]
    server_url = process_info.get("url")
    
    if not server_url:
        raise HTTPException(
            status_code=500,
            detail=f"Server {server_id} URL not available"
        )
    
    # Create a new session using the SSE transport
    try:
        logger.info(f"Creating new MCP client session for server {server_id} at {server_url}")
        
        # Create session with the SSE client
        read_stream, write_stream = await asyncio.to_thread(
            lambda: asyncio.run(sse_client_wrapper(server_url))
        )
        
        session = ClientSession(read_stream, write_stream)
        
        # Initialize the session
        await session.initialize()
        
        # Cache the session
        mcp_sessions[server_id] = {
            "session": session,
            "valid": True,
            "created_at": time.time()
        }
        
        return session
    except Exception as e:
        logger.error(f"Error creating MCP client session for server {server_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error connecting to MCP server: {str(e)}"
        )

# Helper to run sse_client in a synchronous context
async def sse_client_wrapper(url):
    async with sse_client(url) as (read_stream, write_stream):
        return read_stream, write_stream

# API endpoints
@app.get("/servers", response_model=Dict[str, Any])
async def list_servers(api_key: str = Depends(verify_api_key)):
    """
    List all configured MCP servers.
    This endpoint returns information about all servers defined in the servers.json file.
    """
    server_configs = load_server_configs()
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
        "mcp_sdk_version": mcp_version
    }

@app.get("/servers/{server_id}/tools", response_model=List[Dict[str, Any]])
async def list_tools(
    server_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    List the tools available from an MCP server.
    Uses the MCP protocol to communicate with the server.
    """
    try:
        session = await get_mcp_client_session(server_id)
        
        logger.info(f"Requesting tool list from MCP server {server_id}")
        tools = await session.list_tools()
        
        # Format the tools for external consumption
        formatted_tools = []
        for tool in tools:
            formatted_tools.append({
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters"),
                "server_id": server_id
            })
        
        return formatted_tools
    except Exception as e:
        logger.error(f"Error listing tools from MCP server {server_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing tools: {str(e)}"
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
        result = await session.call_tool(
            name=tool_call.tool_name,
            arguments=tool_call.tool_args,
            call_id=call_id
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
    server_configs = load_server_configs()
    configured_count = len(server_configs.get("mcpServers", {}))
    
    # Count running servers
    running_count = 0
    for server_id, process_info in mcp_processes.items():
        if process_info.get("process") and process_info.get("process").poll() is None:
            running_count += 1
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "mcp_sdk_version": mcp_version,
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