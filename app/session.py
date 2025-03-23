"""
Session management for the n8n-mcp-bridge application.
Handles MCP client sessions and communication with MCP servers.
"""

import logging
import uuid
import time
from typing import Dict, Any, Optional

# Import the MCP SDK
try:
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp import StdioServerParameters
except ImportError:
    logger = logging.getLogger('n8n_mcp_bridge.session')
    logger.warning("MCP SDK not found, installing...")
    import subprocess
    subprocess.run(["pip", "install", "mcp"], check=True)
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp import StdioServerParameters

# Set up logging
logger = logging.getLogger('n8n_mcp_bridge.session')

# Global state
mcp_sessions = {}  # Store active MCP client sessions


def get_mcp_client_session(server_id: str, mcp_processes: Dict[str, Any]) -> Optional[ClientSession]:
    """
    Get or create an MCP client session for a server.
    Returns a session if successful, None otherwise.
    """
    # Check if we already have a session for this server
    if server_id in mcp_sessions:
        # Check if the session is still active
        session = mcp_sessions[server_id]
        if hasattr(session, 'is_closed') and not session.is_closed:
            return session
    
    # No active session found, create a new one
    if server_id not in mcp_processes:
        logger.error(f"MCP server {server_id} not found in active processes")
        return None
    
    try:
        process_info = mcp_processes[server_id]
        process = process_info["process"]
        
        # Create an MCP client session
        # By default, we'll use stdio for communication (most reliable)
        params = StdioServerParameters(stdin=process.stdin, stdout=process.stdout)
        session = stdio_client(params)
        
        # Wait for the session to be ready
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Test if session is active by listing tools
                tools = session.list_tools()
                if tools:
                    # Session is active, store and return it
                    mcp_sessions[server_id] = session
                    logger.info(f"Created MCP client session for server {server_id}")
                    return session
            except Exception as e:
                logger.warning(f"Failed to connect to server {server_id}, retrying... ({str(e)})")
                retry_count += 1
                time.sleep(1)
        
        logger.error(f"Failed to create MCP client session for server {server_id} after {max_retries} retries")
        return None
    
    except Exception as e:
        logger.error(f"Error creating MCP client session for server {server_id}: {str(e)}")
        return None


def close_mcp_session(server_id: str) -> bool:
    """
    Close an MCP client session.
    Returns True if successful, False otherwise.
    """
    if server_id not in mcp_sessions:
        logger.warning(f"MCP session for server {server_id} not found")
        return False
    
    try:
        session = mcp_sessions[server_id]
        session.close()
        del mcp_sessions[server_id]
        logger.info(f"Closed MCP client session for server {server_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error closing MCP client session for server {server_id}: {str(e)}")
        return False


def cleanup_sessions():
    """Close all MCP client sessions."""
    logger.info("Cleaning up MCP client sessions")
    server_ids = list(mcp_sessions.keys())
    for server_id in server_ids:
        close_mcp_session(server_id)
