"""
Server management for the n8n-mcp-bridge application.
Handles starting, stopping, and monitoring MCP server processes.
"""

import os
import json
import logging
import subprocess
import threading
import signal
import atexit
from typing import Dict, Any, Optional, Set

# Set up logging
logger = logging.getLogger('n8n_mcp_bridge.server')

# Global state
mcp_processes = {}  # Store running MCP server processes
used_ports = set()  # Track used ports for MCP servers


def load_server_configs():
    """
    Load server configurations from the JSON file.
    Returns a dictionary with server configurations or an empty dict if not found.
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "servers.json")
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


def get_next_available_port():
    """Get the next available port for MCP servers."""
    port = 3000
    while port in used_ports:
        port += 1
    used_ports.add(port)
    return port


def find_free_port(start_port: int, end_port: int) -> Optional[int]:
    """Find an available port in the given range."""
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


def start_mcp_server(server_id: str, server_config: Dict[str, Any]):
    """
    Start an MCP server process.
    Returns process information if successful, None otherwise.
    """
    try:
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Prepare command
        cmd = [server_config["command"]]
        cmd.extend(server_config.get("args", []))
        
        # Get a unique port for this server
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
        stdout_log = open(f"{logs_dir}/{server_id}_stdout.log", "w")
        stderr_log = open(f"{logs_dir}/{server_id}_stderr.log", "w")
        
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
            "config": server_config,
            "port": port,
            "logs": {
                "stdout": stdout_log,
                "stderr": stderr_log
            },
            "threads": {
                "stdout": stdout_thread
            }
        }
        
        logger.info(f"MCP server {server_id} started with PID {process.pid}")
        return mcp_processes[server_id]
    
    except Exception as e:
        logger.error(f"Error starting MCP server {server_id}: {str(e)}")
        return None


def stop_mcp_server(server_id: str):
    """
    Stop an MCP server process.
    Returns True if successful, False otherwise.
    """
    if server_id not in mcp_processes:
        logger.warning(f"MCP server {server_id} not found")
        return False
    
    try:
        process_info = mcp_processes[server_id]
        process = process_info["process"]
        
        logger.info(f"Stopping MCP server {server_id} with PID {process.pid}")
        
        # On macOS/Linux use SIGTERM, then SIGKILL if it doesn't work
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"MCP server {server_id} did not terminate gracefully, using SIGKILL")
            process.kill()
        
        # Close log files
        for log_file in process_info["logs"].values():
            if log_file:
                log_file.close()
        
        # Remove from processes dict
        used_ports.discard(process_info["port"])
        del mcp_processes[server_id]
        
        logger.info(f"MCP server {server_id} stopped")
        return True
    
    except Exception as e:
        logger.error(f"Error stopping MCP server {server_id}: {str(e)}")
        return False


def cleanup_processes():
    """Stop all MCP server processes on shutdown."""
    logger.info("Cleaning up MCP server processes")
    server_ids = list(mcp_processes.keys())
    for server_id in server_ids:
        stop_mcp_server(server_id)


def start_configured_servers():
    """Start all MCP servers defined in the configuration."""
    server_configs = load_server_configs()
    for server_id, config in server_configs.get("mcpServers", {}).items():
        start_mcp_server(server_id, config)


# Register cleanup handler for graceful shutdown
atexit.register(cleanup_processes)

# Register signal handlers
signal.signal(signal.SIGINT, lambda sig, frame: (cleanup_processes(), exit(0)))
signal.signal(signal.SIGTERM, lambda sig, frame: (cleanup_processes(), exit(0)))
