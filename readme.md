# n8n-mcp-bridge

A lightweight bridge service that connects n8n workflows to Model Context Protocol (MCP) servers. This enables AI agents in n8n to seamlessly leverage specialized tools from MCP servers without requiring self-hosting of n8n.

## What this service does

The n8n-mcp-bridge provides three core functionalities:

1. **List Available MCP Servers**: Returns a list of all configured MCP servers with their descriptions, allowing an AI agent in n8n to discover which specialized tools are available.

2. **Discover Tools from an MCP Server**: Queries a specific MCP server to list all available tools and their schemas, following the MCP protocol specification.

3. **Execute Tool Calls**: Allows executing specific tools on an MCP server and returning the results, making external functionality available to n8n workflows.

## Project Structure

```
/n8n-mcp-bridge/
├── main.py                  # Main application file containing all functionality
├── test_api.py              # Test script for API endpoints
├── config/                  # Configuration files
│   └── servers.json         # Server configurations
├── logs/                    # Log directory (generated at runtime)
├── requirements.txt         # Python dependencies
└── README.md                # Documentation
```

## How it works

This service:

1. **Spawns Local MCP Servers**: Automatically downloads and runs Node.js-based MCP servers locally using the configurations defined in `config/servers.json`.

2. **Translates HTTP to MCP Protocol**: Converts standard HTTP requests from n8n into the proper MCP protocol format.

3. **Manages Server Lifecycle**: Handles starting, monitoring, and gracefully shutting down MCP server processes.

4. **Secures Access**: Provides API key authentication to ensure only authorized clients can access the service.

## Pre-configured MCP Servers

This service comes pre-configured with the following MCP servers:

1. **Todoist** (@abhiz123/todoist-mcp-server): Interact with Todoist tasks and projects
2. **Memory** (@modelcontextprotocol/server-memory): Store and retrieve information in a persistent memory store
3. **Slack** (@modelcontextprotocol/server-slack): Send and manage messages in Slack
4. **Brave Search** (@modelcontextprotocol/server-brave-search): Perform web searches using Brave Search

## Configuration

### MCP Server Configuration

MCP servers are configured in the `config/servers.json` file with the following structure:

```json
{
  "mcpServers": {
    "server-id": {
      "command": "npx",
      "args": ["@package/name"],
      "env": {
        "API_KEY": "your-api-key"
      }
    }
  }
}
```

### Environment Variables

Required environment variables are stored in the `.env` file:

```
X_API_KEY=your-service-api-key
ALLOWED_ORIGINS=https://your-n8n-instance.example.com,http://localhost:5678
BRAVE_API_KEY=your-brave-search-api-key
SLACK_BOT_TOKEN=your-slack-bot-token
SLACK_TEAM_ID=your-slack-team-id
TODOIST_API_TOKEN=your-todoist-api-token
```

- `X_API_KEY`: Authentication key for accessing this service
- `ALLOWED_ORIGINS`: Comma-separated list of domains allowed to make cross-origin requests to this service (required when deployed to production with n8n)
- `BRAVE_API_KEY`: API key for Brave Search MCP server
- `SLACK_BOT_TOKEN` & `SLACK_TEAM_ID`: Credentials for Slack MCP server
- `TODOIST_API_TOKEN`: API token for Todoist MCP server

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the service and information about configured servers.

### List All MCP Servers
```
GET /servers
```
Returns a list of all configured MCP servers with their status and metadata.

### Discover Tools from an MCP Server
```
GET /server/{server_id}/tools
```
Returns a list of all tools available from the specified MCP server, including their schemas.

### Call an MCP Server
```
POST /call/{server_id}
```
Sends a full MCP protocol request to the specified server.

### Make a Tool Call
```
POST /tool-call/{server_id}
```
Simplified endpoint to execute a single tool call on the specified MCP server.

## Deployment on Render

### 1. Create a new Web Service

1. Go to the Render dashboard and click "New" > "Web Service"
2. Connect your GitHub repository
3. Choose a name for your service
4. Select "Python 3" as the runtime
5. Set the build command: `pip install -r requirements.txt`
6. Set the start command: `python main.py`

### 2. Set Environment Variables

Add the following environment variables:

- `X_API_KEY`: Your secure API key for authentication
- `NODE_VERSION`: `16` or higher (to ensure Node.js is available)
- `NPM_CONFIG_PRODUCTION`: `false` (to ensure dev dependencies are installed)

### 3. Add Service-Specific API Keys

Add the necessary API keys for the MCP servers you want to use:

- `BRAVE_API_KEY`: For Brave Search
- `SLACK_BOT_TOKEN` and `SLACK_TEAM_ID`: For Slack integration
- `TODOIST_API_TOKEN`: For Todoist integration

## Using with n8n

### 1. Test the service is running

Send a GET request to `/health` to verify the service is running.

### 2. List available MCP servers

Send a GET request to `/servers` with your API key in the `X-API-Key` header.

### 3. Discover tools from a server

Send a GET request to `/server/{server_id}/tools` with your API key in the `X-API-Key` header.

### 4. Make tool calls

In your n8n workflow, add an HTTP Request node with:

- Method: POST
- URL: `https://your-render-service.onrender.com/tool-call/{server_id}`
- Headers:
  - `X-API-Key`: `your-api-key`
  - `Content-Type`: `application/json`
- Body:
  ```json
  {
    "tool_name": "example_tool",
    "tool_args": {
      "arg1": "value1"
    },
    "tool_call_id": "unique-id"
  }
  ```

## Development

### Setting up the development environment

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with the necessary API keys
6. Run the service: `python main.py`
