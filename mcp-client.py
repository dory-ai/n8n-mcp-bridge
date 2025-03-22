import os
import json
import logging
import requests
import uvicorn
import subprocess
import time
import signal
import atexit
import uuid
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Header, Body, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import threading

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

app = FastAPI(
    title="MCP Client Service",
    description="A service that acts as a bridge between n8n and MCP servers",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication
API_KEY = os.getenv("API_KEY", "default-api-key")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
        )
    return api_key

# MCP Models
class MCPToolCall(BaseModel):
    tool_name: str
    tool_args: Dict[str, Any]
    tool_call_id: str

class MCPRequest(BaseModel):
    messages: List[Dict[str, Any]]
    tool_calls: List[MCPToolCall]
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MCPToolResponse(BaseModel):
    tool_call_id: str
    response: Dict[str, Any]

class MCPConfig(BaseModel):
    server_url: str
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

class MCPServerDefinition(BaseModel):
    package: str
    env: Optional[Dict[str, str]] = None
    args: Optional[List[str]] = None

# Store MCP server configurations
mcp_servers: Dict[str, MCPConfig] = {}

# Store running MCP server processes
mcp_processes: Dict[str, Dict[str, Any]] = {}

# Track used ports to avoid conflicts
used_ports = set()

# Create Node.js server runner script
NODE_RUNNER_SCRIPT = """
const http = require('http');
const { spawn } = require('child_process');
const port = process.env.PORT || 3000;

// Get server configuration from command line arguments
const serverConfig = JSON.parse(process.argv[2]);
const { package, env = {}, args = [] } = serverConfig;

console.log(`Starting MCP server: ${package}`);

// Set environment variables
Object.entries(env).forEach(([key, value]) => {
  process.env[key] = value;
});

// Start the server using npx
const npxProcess = spawn('npx', ['-y', package, ...args], {
  stdio: 'pipe',
  env: { ...process.env }
});

let serverPort = null;
let serverUrl = null;

npxProcess.stdout.on('data', (data) => {
  const output = data.toString();
  console.log(`[${package}] ${output}`);
  
  // Try to extract the port from logs (this may need adjustment based on how the servers log)
  const portMatch = output.match(/listening on (?:port )?(\d+)/i);
  if (portMatch && !serverPort) {
    serverPort = portMatch[1];
    serverUrl = `http://localhost:${serverPort}`;
    console.log(`[${package}] Server detected at ${serverUrl}`);
  }
});

npxProcess.stderr.on('data', (data) => {
  console.error(`[${package}] ${data.toString()}`);
});

npxProcess.on('close', (code) => {
  console.log(`[${package}] process exited with code ${code}`);
  process.exit(code);
});

// Create a proxy HTTP server that will forward requests to the MCP server once it's ready
const server = http.createServer((req, res) => {
  if (req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ 
      status: 'running', 
      package, 
      serverPort,
      serverUrl
    }));
    return;
  }
  
  if (!serverUrl) {
    res.writeHead(503, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ 
      status: 'starting', 
      message: 'MCP server is still starting up' 
    }));
    return;
  }
  
  // Forward the request to the MCP server
  const options = {
    hostname: 'localhost',
    port: serverPort,
    path: req.url,
    method: req.method,
    headers: req.headers
  };
  
  const proxyReq = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
  });
  
  proxyReq.on('error', (e) => {
    console.error(`[Proxy] ${e.message}`);
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ 
      status: 'error', 
      message: `Failed to proxy request: ${e.message}` 
    }));
  });
  
  req.pipe(proxyReq);
});

server.listen(port, () => {
  console.log(`[Proxy] Server running at http://localhost:${port}`);
});

// Handle shutdown
process.on('SIGTERM', () => {
  console.log(`[${package}] Shutting down...`);
  npxProcess.kill();
  server.close();
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log(`[${package}] Shutting down...`);
  npxProcess.kill();
  server.close();
  process.exit(0);
});
"""

# Create the Node.js runner script file
def create_node_runner():
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_node_runner.js")
    with open(script_path, "w") as f:
        f.write(NODE_RUNNER_SCRIPT)
    return script_path

# Start an MCP server as a Node.js process
def start_mcp_server(server_id: str, definition: MCPServerDefinition):
    # Create a unique port for this server
    port = find_free_port(8100, 8999)
    
    # Create a server config for the Node.js runner
    server_config = {
        "package": definition.package,
        "env": definition.env or {},
        "args": definition.args or []
    }
    
    # Serialize the config to pass to Node.js
    config_json = json.dumps(server_config)
    
    # Start the Node.js process
    process = subprocess.Popen(
        ["node", NODE_RUNNER_PATH, config_json],
        env={**os.environ, "PORT": str(port)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    logger.info(f"Started MCP server '{server_id}' (package: {definition.package}) on port {port}")
    
    # Store the process info
    mcp_processes[server_id] = {
        "process": process,
        "port": port,
        "url": f"http://localhost:{port}",
        "definition": definition.dict(),
        "started_at": time.time()
    }
    
    # Give the server a moment to start up
    time.sleep(2)
    
    # Add it to the server configurations
    mcp_servers[server_id] = MCPConfig(
        server_url=f"http://localhost:{port}"
    )
    
    return port

# Start a server using Claude Desktop format configuration
def start_server_claude_format(server_id: str, config: Dict[str, Any]):
    if "command" not in config or not config["command"]:
        logger.warning(f"Missing command for server '{server_id}'")
        return
    
    command = config["command"]
    args = config.get("args", [])
    env_vars = config.get("env", {})
    
    # Check for required environment variables
    if server_id == "slack" and (not env_vars.get("SLACK_BOT_TOKEN") or not env_vars.get("SLACK_TEAM_ID")):
        logger.warning(f"Skipping '{server_id}' server: Missing required environment variables (SLACK_BOT_TOKEN, SLACK_TEAM_ID)")
        return
    
    if server_id == "brave-search" and not env_vars.get("BRAVE_API_KEY"):
        logger.warning(f"Skipping '{server_id}' server: Missing required environment variable (BRAVE_API_KEY)")
        return
    
    if server_id == "todoist" and not env_vars.get("TODOIST_API_TOKEN"):
        logger.warning(f"Skipping '{server_id}' server: Missing required environment variable (TODOIST_API_TOKEN)")
        return
    
    # Get a free port for the server
    start_port = 8100
    end_port = 8999
    server_port = find_free_port(start_port, end_port)
    if not server_port:
        raise Exception(f"No free ports available in range {start_port}-{end_port}")
    
    logger.info(f"Starting MCP server '{server_id}' (command: {command}) on port {server_port}")
    
    # For npx servers, we can use the existing NODE_RUNNER_SCRIPT
    if command == "npx":
        # Extract the package name (usually the second argument after -y)
        package = ""
        remaining_args = []
        
        for i, arg in enumerate(args):
            if arg == "-y" and i+1 < len(args):
                package = args[i+1]
                remaining_args = args[i+2:] if i+2 < len(args) else []
                break
        
        if not package:
            logger.warning(f"Could not find package name in args for '{server_id}'")
            return
        
        # Use the existing Node.js runner for npx packages
        server_config = {
            "package": package,
            "env": env_vars,
            "args": remaining_args
        }
        
        node_script_path = NODE_RUNNER_PATH
        if not node_script_path:
            node_script_path = create_node_runner()
        
        # Start the Node.js runner process
        process_env = os.environ.copy()
        process_env.update({"PORT": str(server_port)})
        process_env.update(env_vars)
        
        process = subprocess.Popen(
            ["node", node_script_path, json.dumps(server_config)],
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
    else:
        # For other commands, spawn process directly
        process_env = os.environ.copy()
        process_env.update(env_vars)
        
        cmd = [command] + args
        process = subprocess.Popen(
            cmd,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
    
    # Store information about the server
    mcp_processes[server_id] = {
        "process": process,
        "port": server_port,
        "command": command,
        "args": args,
        "env": env_vars
    }
    
    # Configure the server in our registry
    mcp_servers[server_id] = MCPConfig(
        server_url=f"http://localhost:{server_port}"
    )
    
    # Start a thread to monitor the process output
    def monitor_output():
        for line in process.stdout:
            print(f"[{server_id}] {line.strip()}")
    
    threading.Thread(target=monitor_output, daemon=True).start()
    
    return server_port

# Find a free port to use
def find_free_port(start_port, end_port):
    import socket
    
    # Avoid reusing ports that are already allocated
    for port in range(start_port, end_port + 1):
        if port in used_ports:
            continue
            
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                used_ports.add(port)  # Mark this port as used
                return port
            except OSError:
                continue
    
    return None

# Clean up processes on shutdown
def cleanup_processes():
    logger.info("Cleaning up MCP server processes...")
    for server_id, info in mcp_processes.items():
        try:
            if "process" in info and info["process"].poll() is None:
                logger.info(f"Stopping MCP server '{server_id}'")
                info["process"].terminate()
                info["process"].wait(timeout=5)
        except Exception as e:
            logger.error(f"Error stopping MCP server '{server_id}': {str(e)}")
    
    # Clear the used ports set
    used_ports.clear()

atexit.register(cleanup_processes)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, lambda sig, frame: (cleanup_processes(), exit(0)))
signal.signal(signal.SIGTERM, lambda sig, frame: (cleanup_processes(), exit(0)))

# API Endpoints

@app.get("/servers", tags=["MCP Server Management"])
async def list_servers(api_key: str = Depends(verify_api_key)):
    """
    List all configured MCP servers.
    
    Returns:
    - local_servers: List of MCP servers running locally
    - external_servers: List of external MCP servers configured
    """
    result = {
        "local_servers": [],
        "external_servers": []
    }
    
    for server_id, config in mcp_servers.items():
        if server_id in mcp_processes:
            # This is a local server
            info = mcp_processes[server_id]
            result["local_servers"].append({
                "id": server_id,
                "package": info.get("definition", {}).get("package", ""),
                "url": info.get("url", ""),
                "port": info.get("port", 0),
                "running": info["process"].poll() is None,
                "started_at": info.get("started_at", 0)
            })
        else:
            # This is an external server
            result["external_servers"].append({
                "id": server_id,
                "url": config.server_url
            })
    
    return result

@app.post("/call/{server_id}", tags=["MCP Operations"])
async def call_mcp_server(
    server_id: str,
    request_body: Dict[str, Any] = Body(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Call an MCP server with the provided request body.
    
    The request body should follow the MCP specification and include:
    - messages: List of message objects
    - tool_calls: List of tool call objects
    
    Parameters:
    - server_id: ID of the MCP server to call
    
    Returns:
    The response from the MCP server
    """
    if server_id not in mcp_servers:
        raise HTTPException(
            status_code=404,
            detail=f"MCP server '{server_id}' not found",
        )
    
    server_config = mcp_servers[server_id]
    server_url = server_config.server_url
    
    headers = {
        "Content-Type": "application/json"
    }
    
    if server_config.api_key:
        headers["X-API-Key"] = server_config.api_key
    
    if server_config.headers:
        headers.update(server_config.headers)
    
    try:
        response = requests.post(
            f"{server_url}/v1",
            json=request_body,
            headers=headers
        )
        
        return response.json()
    except Exception as e:
        logger.error(f"Error calling MCP server '{server_id}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error calling MCP server: {str(e)}",
        )

@app.post("/tool-call/{server_id}", tags=["MCP Operations"])
async def make_tool_call(
    server_id: str,
    tool_call: MCPToolCall,
    api_key: str = Depends(verify_api_key)
):
    """
    Make a single tool call to an MCP server.
    
    This is a simplified version of the /call endpoint that only requires
    a single tool call object instead of a full MCP request.
    
    Parameters:
    - server_id: ID of the MCP server to call
    - tool_call: A single MCPToolCall object containing:
      - tool_name: Name of the tool to call
      - tool_args: Arguments for the tool
      - tool_call_id: Unique ID for the tool call
      
    Returns:
    The tool response from the MCP server
    """
    if server_id not in mcp_servers:
        raise HTTPException(
            status_code=404,
            detail=f"MCP server '{server_id}' not found",
        )
    
    server_config = mcp_servers[server_id]
    server_url = server_config.server_url
    
    # Prepare the MCP request
    mcp_request = {
        "messages": [],
        "tool_calls": [
            {
                "tool_name": tool_call.tool_name,
                "tool_args": tool_call.tool_args,
                "tool_call_id": tool_call.tool_call_id or str(uuid.uuid4())
            }
        ]
    }
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json"
    }
    
    if server_config.api_key:
        headers["X-API-Key"] = server_config.api_key
    
    if server_config.headers:
        headers.update(server_config.headers)
    
    try:
        # Make the request to the MCP server
        response = requests.post(
            f"{server_url}/v1",
            json=mcp_request,
            headers=headers
        )
        
        response_data = response.json()
        
        # Extract the tool response
        if "tool_responses" in response_data:
            for tool_response in response_data["tool_responses"]:
                if tool_response["tool_call_id"] == tool_call.tool_call_id:
                    return {
                        "tool_call_id": tool_response["tool_call_id"],
                        "response": tool_response["response"]
                    }
        
        # Return the full response if the specific tool response can't be found
        return response_data
    except Exception as e:
        logger.error(f"Error making tool call to MCP server '{server_id}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making tool call: {str(e)}",
        )

@app.get("/server/{server_id}/tools", tags=["MCP Operations"])
async def discover_tools(
    server_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Discover the tools available from an MCP server.
    
    This endpoint follows the Model Context Protocol specification for tool discovery.
    It sends a discovery request to the MCP server and returns the available tools
    and their schemas.
    
    Parameters:
    - server_id: ID of the MCP server to discover tools from
    
    Returns:
    A list of available tools and their schemas from the MCP server
    """
    if server_id not in mcp_servers:
        raise HTTPException(
            status_code=404,
            detail=f"MCP server '{server_id}' not found",
        )
    
    server_config = mcp_servers[server_id]
    server_url = server_config.server_url
    
    # Prepare the discovery request following MCP specification
    discovery_request = {
        "messages": [],
        "tool_calls": [
            {
                "tool_name": "__describe_tools",
                "tool_args": {},
                "tool_call_id": str(uuid.uuid4())
            }
        ]
    }
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json"
    }
    
    if server_config.api_key:
        headers["X-API-Key"] = server_config.api_key
    
    if server_config.headers:
        headers.update(server_config.headers)
    
    try:
        # Make the request to the MCP server
        response = requests.post(
            f"{server_url}/v1",
            json=discovery_request,
            headers=headers
        )
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Check if we got a proper tool response
            if "tool_responses" in response_data:
                for tool_response in response_data["tool_responses"]:
                    # Return the first tool response we find
                    return {
                        "status": "success",
                        "tools": tool_response.get("response", {}).get("tools", [])
                    }
            
            # If we didn't get a tool response, return the full response
            return {
                "status": "success",
                "raw_response": response_data
            }
        else:
            # Server doesn't support tool discovery
            # Return a predefined list of tools for known servers as fallback
            known_tools = {
                "todoist": [
                    {"name": "get_tasks", "description": "Get a list of tasks from Todoist"},
                    {"name": "create_task", "description": "Create a new task in Todoist"}
                ],
                "memory": [
                    {"name": "create", "description": "Create a new memory entry"},
                    {"name": "query", "description": "Query existing memories"}
                ],
                "slack": [
                    {"name": "get_channels", "description": "Get a list of channels from Slack"},
                    {"name": "post_message", "description": "Post a message to a Slack channel"}
                ],
                "brave-search": [
                    {"name": "search", "description": "Search the web using Brave Search"}
                ]
            }
            
            return {
                "status": "partial",
                "message": "Server does not support tool discovery. Using predefined tools.",
                "tools": known_tools.get(server_id, [])
            }
    except Exception as e:
        logger.error(f"Error discovering tools for MCP server '{server_id}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error discovering tools: {str(e)}",
        )

@app.get("/health", tags=["System"])
async def health_check():
    """
    Check if the service is running.
    
    Returns:
    - status: Current health status of the service
    - servers: Information about configured servers
    """
    return {
        "status": "healthy",
        "servers": {
            "total": len(mcp_servers),
            "local_running": sum(1 for server_id in mcp_processes if mcp_processes[server_id]["process"].poll() is None)
        }
    }

# Create the Node.js runner script on startup
NODE_RUNNER_PATH = create_node_runner()

# Start pre-configured MCP servers
async def start_preconfigured_servers():
    # Load server configurations from JSON file
    server_configs = load_server_configs()
    
    if not server_configs or "mcpServers" not in server_configs:
        logger.warning("No server configurations found in configuration file")
        # Fall back to default servers if no configurations found
        preconfigured_servers = {
            "todoist": MCPServerDefinition(
                package="@abhiz123/todoist-mcp-server",
                env={}
            ),
            "fetch": MCPServerDefinition(
                package="@modelcontextprotocol/server-fetch"
            ),
            "slack": MCPServerDefinition(
                package="@modelcontextprotocol/server-slack",
                env={}
            ),
            "memory": MCPServerDefinition(
                package="@modelcontextprotocol/server-memory"
            )
        }
        
        # Start each preconfigured server using the old method
        for server_id, definition in preconfigured_servers.items():
            try:
                logger.info(f"Starting preconfigured MCP server: {server_id}")
                start_mcp_server(server_id, definition)
            except Exception as e:
                logger.error(f"Failed to start preconfigured server '{server_id}': {str(e)}")
        
        return
    
    # Start each configured server using the Claude Desktop format
    for server_id, config in server_configs["mcpServers"].items():
        try:
            logger.info(f"Starting configured MCP server: {server_id}")
            start_server_claude_format(server_id, config)
        except Exception as e:
            logger.error(f"Failed to start configured server '{server_id}': {str(e)}")

# Main entry point
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    # Start preconfigured servers
    import asyncio
    asyncio.run(start_preconfigured_servers())
    
    logger.info(f"Starting MCP client service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)