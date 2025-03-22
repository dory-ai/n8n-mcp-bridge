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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcp_client")

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

# Find a free port to use
def find_free_port(start_port, end_port):
    import socket
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(('localhost', port))
            if result != 0:  # Port is available
                return port
    raise RuntimeError("No free ports available")

# Clean up processes on shutdown
def cleanup_processes():
    logger.info("Cleaning up MCP server processes...")
    for server_id, info in mcp_processes.items():
        logger.info(f"Stopping MCP server '{server_id}'")
        try:
            info["process"].terminate()
            info["process"].wait(timeout=5)
        except Exception as e:
            logger.error(f"Error stopping MCP server '{server_id}': {str(e)}")
            try:
                info["process"].kill()
            except:
                pass

atexit.register(cleanup_processes)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, lambda sig, frame: (cleanup_processes(), exit(0)))
signal.signal(signal.SIGTERM, lambda sig, frame: (cleanup_processes(), exit(0)))

# Routes
@app.post("/configure")
async def configure_mcp_server(
    server_id: str, 
    config: MCPConfig,
    api_key: str = Depends(verify_api_key)
):
    """Configure an external MCP server connection"""
    mcp_servers[server_id] = config
    return {"status": "success", "message": f"MCP server {server_id} configured successfully"}

@app.post("/configure-local")
async def configure_local_mcp_server(
    server_id: str,
    definition: MCPServerDefinition,
    api_key: str = Depends(verify_api_key)
):
    """Configure and start a local MCP server from an npm package"""
    if server_id in mcp_processes:
        # Server already running, stop it first
        try:
            info = mcp_processes[server_id]
            info["process"].terminate()
            info["process"].wait(timeout=5)
        except Exception as e:
            logger.error(f"Error stopping existing MCP server '{server_id}': {str(e)}")
    
    # Start the new server
    try:
        port = start_mcp_server(server_id, definition)
        return {
            "status": "success",
            "message": f"MCP server {server_id} started successfully",
            "port": port,
            "url": f"http://localhost:{port}"
        }
    except Exception as e:
        logger.error(f"Error starting MCP server '{server_id}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error starting MCP server: {str(e)}"
        )

@app.get("/servers")
async def list_servers(api_key: str = Depends(verify_api_key)):
    """List all configured MCP servers"""
    result = {
        "external_servers": [],
        "local_servers": []
    }
    
    for server_id, config in mcp_servers.items():
        if server_id in mcp_processes:
            # This is a local server
            info = mcp_processes[server_id]
            result["local_servers"].append({
                "id": server_id,
                "package": info["definition"]["package"],
                "url": info["url"],
                "running": info["process"].poll() is None,
                "started_at": info["started_at"]
            })
        else:
            # This is an external server
            result["external_servers"].append({
                "id": server_id,
                "url": config.server_url
            })
    
    return result

@app.post("/call/{server_id}")
async def call_mcp_server(
    server_id: str,
    request_body: Dict[str, Any] = Body(...),
    api_key: str = Depends(verify_api_key)
):
    """Call an MCP server with the provided request body"""
    if server_id not in mcp_servers:
        raise HTTPException(
            status_code=404,
            detail=f"MCP server {server_id} not found. Please configure it first."
        )
    
    server_config = mcp_servers[server_id]
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add API key if provided in config
    if server_config.api_key:
        headers["Authorization"] = f"Bearer {server_config.api_key}"
    
    # Add any additional headers from config
    if server_config.headers:
        headers.update(server_config.headers)
    
    try:
        logger.info(f"Sending request to MCP server {server_id}")
        response = requests.post(
            server_config.server_url,
            json=request_body,
            headers=headers
        )
        
        # Log response status
        logger.info(f"MCP server responded with status {response.status_code}")
        
        # Check if response is successful
        response.raise_for_status()
        
        # Return the response from the MCP server
        return response.json()
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling MCP server: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error calling MCP server: {str(e)}"
        )

@app.post("/tool-call/{server_id}")
async def make_tool_call(
    server_id: str,
    tool_call: MCPToolCall,
    api_key: str = Depends(verify_api_key)
):
    """Make a single tool call to an MCP server"""
    if server_id not in mcp_servers:
        raise HTTPException(
            status_code=404,
            detail=f"MCP server {server_id} not found. Please configure it first."
        )
    
    server_config = mcp_servers[server_id]
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add API key if provided in config
    if server_config.api_key:
        headers["Authorization"] = f"Bearer {server_config.api_key}"
    
    # Add any additional headers from config
    if server_config.headers:
        headers.update(server_config.headers)
    
    # Construct MCP request for a tool call
    mcp_request = {
        "tool_calls": [
            {
                "tool_name": tool_call.tool_name,
                "tool_args": tool_call.tool_args,
                "tool_call_id": tool_call.tool_call_id
            }
        ]
    }
    
    try:
        logger.info(f"Making tool call {tool_call.tool_name} to MCP server {server_id}")
        response = requests.post(
            server_config.server_url,
            json=mcp_request,
            headers=headers
        )
        
        # Log response status
        logger.info(f"MCP server responded with status {response.status_code}")
        
        # Check if response is successful
        response.raise_for_status()
        
        # Return the response from the MCP server
        return response.json()
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making tool call: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making tool call: {str(e)}"
        )

@app.delete("/servers/{server_id}")
async def delete_server(
    server_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Delete an MCP server configuration and stop it if it's running locally"""
    if server_id not in mcp_servers:
        raise HTTPException(
            status_code=404,
            detail=f"MCP server {server_id} not found"
        )
    
    # If it's a local server, stop it
    if server_id in mcp_processes:
        try:
            info = mcp_processes[server_id]
            info["process"].terminate()
            info["process"].wait(timeout=5)
            del mcp_processes[server_id]
        except Exception as e:
            logger.error(f"Error stopping MCP server '{server_id}': {str(e)}")
    
    # Remove the configuration
    del mcp_servers[server_id]
    
    return {"status": "success", "message": f"MCP server {server_id} deleted successfully"}

@app.get("/health")
async def health_check():
    """Check if the service is running"""
    return {
        "status": "healthy",
        "servers": {
            "total": len(mcp_servers),
            "local_running": sum(1 for info in mcp_processes.values() if info["process"].poll() is None)
        }
    }

# Create the Node.js runner script on startup
NODE_RUNNER_PATH = create_node_runner()

# Start pre-configured MCP servers
async def start_preconfigured_servers():
    # Define the servers you want to pre-configure
    preconfigured_servers = {
        "todoist": MCPServerDefinition(
            package="@abhiz123/todoist-mcp-server",
            env={}  # You'll need to add TODOIST_API_TOKEN in production
        ),
        "fetch": MCPServerDefinition(
            package="@modelcontextprotocol/server-fetch"
        ),
        "slack": MCPServerDefinition(
            package="@modelcontextprotocol/server-slack",
            env={}  # You'll need to add SLACK_TOKEN in production
        ),
        "memory": MCPServerDefinition(
            package="@modelcontextprotocol/server-memory"
        )
    }
    
    # Start each preconfigured server
    for server_id, definition in preconfigured_servers.items():
        try:
            logger.info(f"Starting preconfigured MCP server: {server_id}")
            start_mcp_server(server_id, definition)
        except Exception as e:
            logger.error(f"Failed to start preconfigured server '{server_id}': {str(e)}")

# Main entry point
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    # Start preconfigured servers
    import asyncio
    asyncio.run(start_preconfigured_servers())
    
    logger.info(f"Starting MCP client service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)