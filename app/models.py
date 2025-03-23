"""
Data models for the n8n-mcp-bridge application.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class MCPToolCall(BaseModel):
    """Model for MCP tool call requests."""
    tool_name: str
    tool_args: Dict[str, Any]
    tool_call_id: Optional[str] = None


class MCPServerDefinition(BaseModel):
    """Model for MCP server configuration."""
    package: str
    description: Optional[str] = "MCP Server"
    env: Optional[Dict[str, str]] = None
    args: Optional[List[str]] = None
