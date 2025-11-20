# LFM2 SDK

The LFM2 SDK contains a comprehensive collection of tools and demonstrations for working with Large Foundation Models, particularly for tool calling, MCP integration, and model evaluation. This SDK provides various proof-of-concept implementations and test suites to validate LLM capabilities.

## Contents

### Tool Chain and Orchestration

- **continous_orchestrator.py** - Production Tool Chain Orchestrator that runs continuously with discovered limits in mind
  - Manages conversation state with limit awareness
  - Implements history compression and tool result caching
  - Supports continuous operation with chain depth management

- **tool_chain_orchestrator.py** - Tool chain orchestrator with limit awareness
  - Manages tool chains and conversation history
  - Implements depth and history retention limits

### Testing and Evaluation

- **cycles.py** - Tool Calling Test Framework with clean architecture
  - Tests single, multi-turn, and sequential tool calls
  - Validates model's use of tool results
  - Provides comprehensive test suite

- **test_chain_limits.py** - Tests chain depth and history retention limits
  - Evaluates how deep tool chains can go before breaking
  - Tests recall of earlier results in conversation history

- **test_limits.py** - Three-axis testing of tool calling limits
  - Tests parameter scaling, multiple tools, and sequential calls
  - Determines operational limits across different dimensions

- **test_tools.py** - Basic tool calling API test suite with correct format
  - Tests basic, visibility, sequential, and multi-turn scenarios
  - Uses proper OpenAI nested format for tools

- **test_tools_verbose.py** - Enhanced test suite with full response logging
  - Provides complete server request/response logging
  - Offers detailed performance metrics

- **test_tools_verbose2.py** - Corrected multi-turn tests with tool execution
  - Includes proper mock tool execution
  - Tests context usage between turns

### Tool Format Testing

- **test_tool_foramt_for_mcp.py** - Tests correct tool format for MCP
  - Validates LFM2's tool calling with nested format
  - Tests specific tool call markers

- **test_tool_format.py** - Deep diagnostic for various tool formats
  - Tests 8 different format hypotheses
  - Identifies which format is accepted by the server

- **test_toolcall.py** - Initial tool calling API test suite
  - Tests basic tool invocation and visibility
  - Includes multi-turn conversation tests

### MCP Integration

- **mcp_chain_discovery.py** - Chain depth discovery with real MCP operations
  - Tests actual chain depth limits with real tool calls
  - Discovers operational limits using MCP filesystem

- **mcp_chat.py** - Interactive chat with MCP filesystem tools
  - Provides real-time chat interface with filesystem tools
  - Includes anti-loop protection and conversation management

- **mcp_chat_enhanced.py** - Enhanced tool transparency chat
  - Shows every step of tool execution process
  - Provides color-coded output for different stages

- **minimal_mcp.py** - Minimal MCP example with directory operations
  - Demonstrates basic MCP integration
  - Tests directory listing, file creation, and reading operations

### Research and Analysis

- **deep_researcher.py** - Brave Search API integration for deep research
  - Performs multi-phase research using real APIs
  - Includes fact extraction, cross-validation, and synthesis

## Key Features

- **Tool Calling**: Comprehensive testing and implementation of tool calling capabilities
- **MCP Integration**: Real Model Context Protocol integration with filesystem operations
- **Chain Management**: Sophisticated handling of tool chain depth and history retention
- **Format Compatibility**: Correct OpenAI nested format implementation
- **Real API Integration**: Integration with real services like Brave Search
- **Multi-turn Conversations**: Support for context-aware conversation flows
- **Performance Metrics**: Detailed timing and usage information

## Configuration

Most scripts in this SDK expect:
- Local LLM endpoint at `http://localhost:8080/v1`
- Models like `LFM2-8B-A1B-BF16-cuda` or `LFM2-1.2B-Tool-Q4_K_M-cuda`
- MCP filesystem server for MCP-related scripts
- Brave Search API key for deep_researcher.py

## Usage

Each script can be run independently to test specific functionality or demonstrate different aspects of LLM tool calling. The scripts range from simple tests to full-featured applications, making them suitable for both development and production use.

## Dependencies

- `requests` for API communication
- `json` for data serialization
- `typing` for type hints
- `re` for pattern matching
- Model Context Protocol (MCP) server for MCP scripts
- Local LLM server supporting tool calls