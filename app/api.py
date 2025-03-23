"""
API endpoints for the n8n-mcp-bridge application.
"""

import os
import logging
import json
import uuid
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Body, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

from .models import MCPToolCall, MCPServerDefinition
from .server import mcp_processes, start_mcp_server, stop_mcp_server, load_server_configs
from .session import get_mcp_client_session, close_mcp_session

# Set up logging
logger = logging.getLogger('n8n_mcp_bridge.api')

# Create FastAPI app
app = FastAPI(
    title="n8n-mcp-bridge",
    description="A lightweight bridge service that connects n8n workflows to Model Context Protocol (MCP) servers",
    version="1.0.0",
)

# CORS middleware configured with environment variables
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
    """Verify that the API key is valid."""
    if api_key != X_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
        )
    return api_key


@app.get("/servers", summary="List all configured MCP servers")
async def list_servers(api_key: str = Depends(verify_api_key)):
    """
    List all configured MCP servers.
    This endpoint returns information about all servers defined in the servers.json file.
    """
    try:
        server_configs = load_server_configs()
        
        # Enhance the server configs with status information
        servers = {}
        for server_id, config in server_configs.get("mcpServers", {}).items():
            # Check if server is running
            is_running = server_id in mcp_processes
            
            # Get process info if available
            process_info = mcp_processes.get(server_id, {})
            
            servers[server_id] = {
                "id": server_id,
                "name": config.get("name", server_id),
                "description": config.get("description", ""),
                "package": config.get("package", ""),
                "status": "running" if is_running else "stopped",
                "pid": process_info.get("process", {}).pid if is_running else None,
            }
        
        return {"servers": servers}
    
    except Exception as e:
        logger.error(f"Error listing servers: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing servers: {str(e)}",
        )


@app.get("/server/{server_id}/tools", summary="List available tools for an MCP server")
async def list_server_tools(server_id: str, api_key: str = Depends(verify_api_key)):
    """
    List available tools for an MCP server.
    """
    try:
        # Check if server is configured
        server_configs = load_server_configs()
        if server_id not in server_configs.get("mcpServers", {}):
            raise HTTPException(
                status_code=404,
                detail=f"MCP server {server_id} not found in configuration",
            )
        
        # Get server config
        server_config = server_configs["mcpServers"][server_id]
        
        # Check if server is running, start it if not
        if server_id not in mcp_processes:
            logger.info(f"Starting MCP server {server_id} for tool discovery")
            process_info = start_mcp_server(server_id, server_config)
            if not process_info:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to start MCP server {server_id}",
                )
        
        # Get MCP client session
        session = get_mcp_client_session(server_id, mcp_processes)
        if not session:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create MCP client session for server {server_id}",
            )
        
        # List tools
        tools = session.list_tools()
        
        # Format response
        tool_list = []
        for tool_name, tool_info in tools.items():
            # Extract schema from tool info
            schema = tool_info.get("schema", {})
            
            # Extract description from schema
            description = schema.get("description", "")
            
            # Extract parameters from schema
            parameters = {}
            if "properties" in schema:
                for param_name, param_info in schema.get("properties", {}).items():
                    parameters[param_name] = {
                        "type": param_info.get("type", "string"),
                        "description": param_info.get("description", ""),
                        "required": param_name in schema.get("required", []),
                    }
            
            tool_list.append({
                "name": tool_name,
                "description": description,
                "parameters": parameters,
                "schema": schema,
            })
        
        return {
            "server_id": server_id,
            "server_name": server_config.get("name", server_id),
            "tools": tool_list,
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error listing tools for server {server_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing tools for server {server_id}: {str(e)}",
        )


@app.post("/tool-call/{server_id}", summary="Call a tool on an MCP server")
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
        # Check if server is configured
        server_configs = load_server_configs()
        if server_id not in server_configs.get("mcpServers", {}):
            raise HTTPException(
                status_code=404,
                detail=f"MCP server {server_id} not found in configuration",
            )
        
        # Get server config
        server_config = server_configs["mcpServers"][server_id]
        
        # Generate tool_call_id if not provided
        if not tool_call.tool_call_id:
            tool_call.tool_call_id = str(uuid.uuid4())
        
        # Call the tool
        result = await call_server_tool(
            server_id,
            tool_call.tool_name,
            tool_call.tool_args,
            api_key
        )
        
        # Return the result
        return {
            "server_id": server_id,
            "tool_name": tool_call.tool_name,
            "tool_call_id": tool_call.tool_call_id,
            "result": result,
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error calling tool {tool_call.tool_name} on server {server_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error calling tool {tool_call.tool_name} on server {server_id}: {str(e)}",
        )


@app.post("/call/{server_id}", summary="Call an MCP server with a full MCP request")
async def call_server(
    server_id: str,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Call an MCP server with a full MCP request.
    This endpoint is for advanced users who want to send raw MCP protocol requests.
    """
    try:
        # Get request body as raw JSON
        body = await request.json()
        
        # Check if server is configured
        server_configs = load_server_configs()
        if server_id not in server_configs.get("mcpServers", {}):
            raise HTTPException(
                status_code=404,
                detail=f"MCP server {server_id} not found in configuration",
            )
        
        # Get server config
        server_config = server_configs["mcpServers"][server_id]
        
        # Check if server is running, start it if not
        if server_id not in mcp_processes:
            logger.info(f"Starting MCP server {server_id} for call")
            process_info = start_mcp_server(server_id, server_config)
            if not process_info:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to start MCP server {server_id}",
                )
        
        # Get MCP client session
        session = get_mcp_client_session(server_id, mcp_processes)
        if not session:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create MCP client session for server {server_id}",
            )
        
        # Call the MCP server with the raw request
        # This depends on what the request shape is, typically it might use session.call_tool()
        # For now, we'll just pass through the full request
        result = await session.call(body)
        
        # Return the raw result
        return result
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error calling server {server_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error calling server {server_id}: {str(e)}",
        )


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
        # Check if server is configured
        server_configs = load_server_configs()
        if server_id not in server_configs.get("mcpServers", {}):
            raise HTTPException(
                status_code=404,
                detail=f"MCP server {server_id} not found in configuration",
            )
        
        # Get server config
        server_config = server_configs["mcpServers"][server_id]
        
        # Check if server is running, start it if not
        if server_id not in mcp_processes:
            logger.info(f"Starting MCP server {server_id} for tool call")
            process_info = start_mcp_server(server_id, server_config)
            if not process_info:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to start MCP server {server_id}",
                )
        
        # Get MCP client session
        session = get_mcp_client_session(server_id, mcp_processes)
        if not session:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create MCP client session for server {server_id}",
            )
        
        # Call the tool
        result = await session.call_tool(tool_name, tool_args)
        return result
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error calling tool {tool_name} on server {server_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error calling tool {tool_name} on server {server_id}: {str(e)}",
        )


@app.get("/health", summary="Check the health of the service")
async def health_check():
    """
    Check the health of the service.
    """
    try:
        # Get server status
        server_statuses = {}
        server_configs = load_server_configs()
        
        for server_id in server_configs.get("mcpServers", {}):
            is_running = server_id in mcp_processes
            server_statuses[server_id] = {
                "status": "running" if is_running else "stopped",
            }
        
        return {
            "status": "healthy",
            "servers": server_statuses,
            "version": "1.0.0",
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }
