# MCP Chain Discovery

## Overview

MCP Chain Discovery is a tool that discovers actual chain depth limits when using Model Context Protocol (MCP) operations. It tests how deep tool chains can go with real MCP tool calls and evaluates history retention capabilities of the model when processing sequential operations.

## Key Features

- **Real MCP Integration**: Connects to a real MCP filesystem server using subprocess
- **Chain Depth Testing**: Tests progressively deeper tool chains to find breaking points
- **History Retention Testing**: Evaluates how well the model recalls earlier results
- **File System Operations**: Uses real file system operations via MCP (read/write files)
- **Automated Discovery**: Automatically discovers operational limits with real tool calls

## Architecture

### Core Components

- **MCP Server Connection**: Establishes connection to MCP filesystem server via subprocess
- **JSON-RPC Communication**: Uses JSON-RPC protocol to communicate with MCP server
- **Tool Execution**: Executes MCP tools (write_file, read_file) through the protocol
- **Chain Testing**: Builds and tests progressively deeper tool chains
- **History Testing**: Tests recall of earlier operations in the conversation

### Test Procedures

1. **Initialization**: Connects to MCP server and retrieves available tools
2. **Chain Depth Testing**: Creates test files and measures how deep chains can go
3. **History Retention Testing**: Tests recall of earlier file contents after sequential operations

## Configuration

- `LLM_URL`: Local LLM endpoint (default: "http://localhost:8080/v1")
- `MODEL`: Model to use for testing (default: "LFM2-8B-A1B-UD-Q3_K_XL-cpu")
- `MCP_CMD`: Command to start MCP filesystem server (uses @modelcontextprotocol/server-filesystem)
- `max_depth`: Maximum depth to test (default: 15)

## Functionality

### Chain Depth Test
- Creates test files in /tmp directory
- Builds queries that require reading files in sequence
- Tracks how many tool calls are successfully executed
- Determines the maximum reliable chain depth

### History Retention Test
- Creates multiple files with unique data
- Reads all files sequentially
- Tests recall of earlier file contents
- Measures how far back the model can reliably remember results

## Output

The tool provides:
- Progress indicators showing each depth test
- Success/failure status for each depth level
- Count of tools called vs. expected
- Maximum reliable chain depth discovered
- History retention success rate
- Detailed results for each test case

## Limitations

- Requires MCP server to be available
- Uses specific filesystem operations (may not represent all tool types)
- Tests may be affected by rate limiting
- Results may vary based on model performance
- Script terminates MCP process at the end

## Dependencies

- `subprocess` for MCP server communication
- `json` for JSON-RPC message handling
- `requests` for LLM API calls
- `re` for pattern matching
- `time` for rate limiting
- Model Context Protocol (MCP) filesystem server