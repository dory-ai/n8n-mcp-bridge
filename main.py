"""
n8n-mcp-bridge: A lightweight bridge service connecting n8n workflows to MCP servers.

This is the main entry point for the application.
"""

import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('n8n_mcp_bridge')

# Import server management functions
from app.server import start_configured_servers, cleanup_processes

# Import FastAPI app
from app.api import app

# Start configured MCP servers on startup
@app.on_event("startup")
async def startup_event():
    """Start all configured MCP servers on application startup."""
    logger.info("Starting n8n-mcp-bridge service")
    start_configured_servers()

# Clean up processes on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up all MCP server processes on application shutdown."""
    logger.info("Shutting down n8n-mcp-bridge service")
    cleanup_processes()

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    # Start the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="info",
        reload=os.getenv("ENV", "production").lower() == "development"
    )
