"""
Simple test script to verify the MCP client API is working.
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base URL for the API
BASE_URL = "http://localhost:8000"

# API key from environment variables
API_KEY = os.getenv("X_API_KEY")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

def test_health():
    """Test the health check endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.status_code}")
    print(response.json())
    print()

def test_list_servers():
    """Test the list servers endpoint."""
    response = requests.get(f"{BASE_URL}/servers", headers=HEADERS)
    print(f"List servers: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_server_tools(server_id):
    """Test listing tools for a specific server."""
    response = requests.get(f"{BASE_URL}/servers/{server_id}/tools", headers=HEADERS)
    print(f"List tools for {server_id}: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(response.text)
    print()

if __name__ == "__main__":
    # Run the tests
    test_health()
    test_list_servers()
    
    # Test tools for specific servers
    test_server_tools("memory")
    test_server_tools("todoist")
