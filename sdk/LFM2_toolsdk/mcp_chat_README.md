# MCP Chat

## Overview

MCP Chat is an interactive chat interface that connects to a Model Context Protocol (MCP) filesystem server. It provides a real-time chat experience where users can interact with an AI assistant that has access to filesystem tools for reading, writing, listing, and managing files in the /tmp directory.

## Key Features

- **Real-time Interaction**: Provides an interactive command-line interface for chatting with an AI assistant
- **MCP Integration**: Connects to a real MCP filesystem server for actual file operations
- **Filesystem Tools**: Access to 14 different filesystem tools for file management
- **Anti-loop Protection**: Prevents infinite loops by tracking and skipping duplicate tool calls
- **Conversation History**: Maintains conversation history throughout the session
- **Example Mode**: Includes example conversation flow for demonstration purposes

## Architecture

### Core Components

- `MCPChat`: Main class that handles MCP server connection and chat operations
- Tool execution with real MCP protocols (not mocks)
- Conversation history management
- Anti-loop protection mechanism
- Input command processing

### Available Tools

The system provides 14 filesystem tools:
- `read_file`, `read_text_file`, `read_media_file`, `read_multiple_files`: File reading operations
- `write_file`, `edit_file`: File writing and editing operations
- `create_directory`, `list_directory`, `list_directory_with_sizes`, `directory_tree`: Directory operations
- `move_file`, `search_files`, `get_file_info`, `list_allowed_directories`: File management tools

## Configuration

- `LLM_URL`: Local LLM endpoint (default: "http://localhost:8080/v1")
- `MODEL`: Model to use for chat (default: "LFM2-8B-A1B-UD-Q3_K_XL-cpu")
- `MCP_CMD`: Command to start MCP filesystem server (uses @modelcontextprotocol/server-filesystem)

## Usage

### Interactive Mode
Run without arguments to start the interactive chat:
```
python mcp_chat.py
```

### Example Mode
Run with the example argument to see a demonstration:
```
python mcp_chat.py example
```

### Available Commands
- `/help`: Show available tools with descriptions
- `/clear`: Clear conversation history
- `/verbose`: Toggle verbose mode
- `/quit`: Exit the chat

## Command Line

The chat interface provides:
- Real-time interaction with an AI assistant
- File operations through MCP tools
- Color-coded output for different message types
- Command support for chat management

## Limitations

- Requires MCP server to be available
- Working directory restricted to /tmp
- May get stuck in loops despite anti-loop protection
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