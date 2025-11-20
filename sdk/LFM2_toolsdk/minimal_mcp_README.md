# Minimal MCP

## Overview

Minimal MCP is a concise demonstration of Model Context Protocol (MCP) integration with filesystem tools. It shows how to connect to an MCP filesystem server and execute various file operations through an LLM interface. The script runs three distinct tests to demonstrate different capabilities of the MCP integration.

## Key Features

- **Direct MCP Integration**: Connects directly to MCP filesystem server via subprocess
- **Multiple File Operations**: Demonstrates reading, writing, listing, and getting file information
- **Tool Chain Execution**: Shows how multiple tools can be chained in sequence
- **Real File System Access**: Operates on actual files in the /tmp directory
- **JSON-RPC Communication**: Uses JSON-RPC protocol to communicate with MCP server

## Architecture

### Core Components

- MCP server initialization and tool listing
- Conversion of MCP tools to OpenAI format
- Tool execution loop with result handling
- Three distinct test scenarios

### Available Tools

The system provides 14 filesystem tools:
- `read_file`, `read_text_file`, `read_media_file`, `read_multiple_files`: File reading operations
- `write_file`, `edit_file`: File writing and editing operations
- `create_directory`, `list_directory`, `list_directory_with_sizes`, `directory_tree`: Directory operations
- `move_file`, `search_files`, `get_file_info`, `list_allowed_directories`: File management tools

## Tests

### Test 1: Directory Listing
- Lists contents of the /tmp directory
- Demonstrates basic directory operations
- Shows directory and file structure

### Test 2: File Creation and Reading
- Creates a file with specific content
- Reads the file back to verify creation
- Tests write and read operations

### Test 3: Tool Chain Operations
- Chains multiple operations: list, write, read, get file info
- Demonstrates complex multi-step operations
- Shows how multiple tools can work together

## Configuration

- `LLM_URL`: Local LLM endpoint (default: "http://localhost:8080/v1")
- `MODEL`: Model to use for testing (default: "LFM2-8B-A1B-BF16-cuda")
- `MCP_CMD`: Command to start MCP filesystem server (uses @modelcontextprotocol/server-filesystem)

## Output

The script provides clear output for each test:
- MCP tools available
- Step-by-step execution for each test
- Tool call information
- Results of each operation
- Final answers from the LLM

## Limitations

- Requires MCP server to be available
- Operations limited to /tmp directory
- Fixed iteration limit (10) for tool chains
- Depends on the availability of the LLM endpoint
- No error recovery for failed tool calls

## Dependencies

- `subprocess` for MCP server communication
- `json` for JSON-RPC message handling
- `requests` for LLM API calls
- `re` for pattern matching
- Model Context Protocol (MCP) filesystem server