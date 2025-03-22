# Integrated MCP Client Service

This service acts as both an MCP client and a launcher for MCP servers. It can run Node.js-based MCP servers locally and expose them through a unified API, making it easy to use with n8n's HTTP Request nodes.

## Features

- **Run MCP Servers**: Automatically downloads and runs MCP servers from npm
- **API Gateway**: Provides a unified API to communicate with MCP servers
- **Configuration Management**: Manage multiple MCP server configurations
- **Authentication**: Secure API key authentication
- **Process Management**: Handles starting, stopping, and monitoring MCP server processes

## Pre-configured MCP Servers

This service comes pre-configured with the following MCP servers:

1. **Todoist** (@abhiz123/todoist-mcp-server)
2. **Fetch** (@modelcontextprotocol/server-fetch)
3. **Slack** (@modelcontextprotocol/server-slack)
4. **Memory** (@modelcontextprotocol/server-memory)

## Deployment on Render

### 1. Create a new Web Service

1. Go to the Render dashboard and click "New" > "Web Service"
2. Connect your GitHub repository
3. Choose a name for your service
4. Select "Python 3" as the runtime
5. Set the build command: `pip install -r requirements.txt`
6. Set the start command: `python mcp_client_updated.py`

### 2. Set Environment Variables

Add the following environment variables:

- `API_KEY`: Your secure API key for authentication
- `NODE_VERSION`: `16` or higher (to ensure Node.js is available)
- `NPM_CONFIG_PRODUCTION`: `false` (to ensure dev dependencies are installed)

### 3. Add Environment-Specific API Keys (Optional)

For services that require API keys, add them as environment variables:

- `TODOIST_API_TOKEN`: For Todoist integration
- `SLACK_TOKEN`: For Slack integration
- etc.

## Using with n8n

### 1. Test the service is running

Send a GET request to `/health` to verify the service is running.

### 2. List available servers

Send a GET request to `/servers` with your API key in the `X-API-Key` header.

### 3. Make tool calls

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

## API Endpoints

### Health Check
```
GET /health
```

### List All Servers
```
GET /servers
```

### Call an MCP Server
```
POST /call/{server_id}
```

### Make a Tool Call
```
POST /tool-call/{server_id}
```

### Configure an External MCP Server
```
POST /configure?server_id={server_id}
```

### Configure and Start a Local MCP Server
```
POST /configure-local?server_id={server_id}
```

### Delete a Server
```
DELETE /servers/{server_id}
```

## Troubleshooting

If you encounter issues with the Node.js processes not starting:

1. Check the Render logs for any error messages
2. Ensure Node.js 16+ is available in your environment
3. Try restarting the service
4. Make sure the npm packages are accessible

## Security Considerations

- Always use a strong API key
- Consider adding rate limiting in production
- For sensitive integrations, use environment variables for API keys

## Limitations on Render

- Render's free tier has limited RAM and CPU resources
- Services may be put to sleep after inactivity
- Node.js process management can be resource-intensive