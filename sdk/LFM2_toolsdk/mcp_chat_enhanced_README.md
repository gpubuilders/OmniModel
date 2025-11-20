# MCP Chat Enhanced

## Overview

MCP Chat Enhanced is an interactive chat interface that provides full transparency in the tool calling process. It connects to a Model Context Protocol (MCP) filesystem server and shows every step of the tool execution process: from user request to LLM interpretation to tool execution to final response.

## Key Features

- **Full Transparency Mode**: Shows every step of the tool calling process with color-coded output
- **MCP Integration**: Connects to real MCP filesystem server for file operations
- **Interactive Chat**: Provides a command-line chat interface for real-time interaction
- **Anti-Loop Protection**: Prevents infinite loops by tracking duplicate tool calls
- **Command Interface**: Supports special commands for tool listing, clearing history, and debugging

## Architecture

### Core Components

- `MCPChat`: Main class that handles MCP server connection and chat operations
- Tool execution with real MCP protocols (not mocks)
- Conversation history management
- Anti-loop protection mechanism
- Color-coded output for different stages

### Available Tools

The system provides 14 filesystem tools:
- `read_file`, `read_text_file`, `read_media_file`, `read_multiple_files`: File reading operations
- `write_file`, `edit_file`: File writing and editing operations  
- `create_directory`, `list_directory`, `list_directory_with_sizes`, `directory_tree`: Directory operations
- `move_file`, `search_files`, `get_file_info`, `list_allowed_directories`: File management tools

### Tool Flow Transparency

The system shows every step in the process:
1. **User Request**: Shows the original user input
2. **LLM Tool Call**: Shows when the LLM attempts to call a tool
3. **Tool Execution**: Shows the exact tool being executed with arguments
4. **Raw Result**: Shows the unprocessed tool result
5. **Final Response**: Shows the LLM's interpretation of the results

## Configuration

- `LLM_URL`: Local LLM endpoint (default: "http://localhost:8080/v1")
- `MODEL`: Model to use for chat (default: "LFM2-8B-A1B-BF16-cuda")
- `MCP_CMD`: Command to start MCP filesystem server (uses @modelcontextprotocol/server-filesystem)

## Usage

### Interactive Mode
Run without arguments to start the interactive chat:
```
python mcp_chat_enhanced.py
```

### Demonstration Mode
Run with the demo argument to see a complete example:
```
python mcp_chat_enhanced.py demo
```

### Available Commands
- `/help`: Show available tools with descriptions
- `/clear`: Clear conversation history
- `/debug`: Toggle debug mode
- `/verbose`: Toggle verbose mode
- `/quit`: Exit the chat

## Command Line

The chat interface provides real-time interaction where users can:
- Ask questions about files in /tmp directory
- Request file operations (read, write, list, etc.)
- View the complete flow of tool execution
- Access special commands for chat management

## Limitations

- Requires MCP server to be available
- Stops when no more tool calls are detected
- May get stuck in loops if anti-loop protection fails
- Interactive interface requires user input to proceed
- Depends on the availability of the LLM endpoint

## Dependencies

- `subprocess` for MCP server communication
- `json` for JSON-RPC message handling
- `requests` for LLM API calls
- `re` for pattern matching
- `typing` for type hints
- `datetime` for timestamps
- Model Context Protocol (MCP) filesystem server