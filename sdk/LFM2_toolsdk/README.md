# LFM2 SDK

The LFM2 SDK contains a comprehensive collection of tools and demonstrations for working with Large Foundation Models, particularly for tool calling, MCP integration, and model evaluation. This SDK provides various proof-of-concept implementations and test suites to validate LLM capabilities.

## Contents

### Tool Chain and Orchestration

- **[continous_orchestrator.py](continous_orchestrator.py)** - Production Tool Chain Orchestrator that runs continuously with discovered limits in mind
  - Manages conversation state with limit awareness
  - Implements history compression and tool result caching
  - Supports continuous operation with chain depth management
  - [Example Output](logs/continous_orchestrator_output.log)
  - [Documentation](continous_orchestrator_README.md)

- **[tool_chain_orchestrator.py](tool_chain_orchestrator.py)** - Tool chain orchestrator with limit awareness
  - Manages tool chains and conversation history
  - Implements depth and history retention limits
  - [Documentation](tool_chain_orchestrator_README.md)

### Testing and Evaluation

- **[cycles.py](cycles.py)** - Tool Calling Test Framework with clean architecture
  - Tests single, multi-turn, and sequential tool calls
  - Validates model's use of tool results
  - Provides comprehensive test suite
  - [Example Output](logs/cycles_output.log)
  - [Documentation](cycles_README.md)

- **[test_chain_limits.py](test_chain_limits.py)** - Tests chain depth and history retention limits
  - Evaluates how deep tool chains can go before breaking
  - Tests recall of earlier results in conversation history
  - [Example Output](logs/test_chain_limits_output.log)
  - [Documentation](test_chain_limits_README.md)

- **[test_limits.py](test_limits.py)** - Three-axis testing of tool calling limits
  - Tests parameter scaling, multiple tools, and sequential calls
  - Determines operational limits across different dimensions
  - [Example Output](logs/test_limits_output.log)
  - [Documentation](test_limits_README.md)

- **[test_tools.py](test_tools.py)** - Basic tool calling API test suite with correct format
  - Tests basic, visibility, sequential, and multi-turn scenarios
  - Uses proper OpenAI nested format for tools
  - [Example Output](logs/test_tools_output.log)
  - [Documentation](test_tools_README.md)

- **[test_tools_verbose.py](test_tools_verbose.py)** - Enhanced test suite with full response logging
  - Provides complete server request/response logging
  - Offers detailed performance metrics
  - [Example Output](logs/test_tools_verbose_output.log)
  - [Documentation](test_tools_verbose_README.md)

- **[test_tools_verbose2.py](test_tools_verbose2.py)** - Corrected multi-turn tests with tool execution
  - Includes proper mock tool execution
  - Tests context usage between turns
  - [Example Output](logs/test_tools_verbose2_output.log)
  - [Documentation](test_tools_verbose2_README.md)

### Tool Format Testing

- **[test_tool_foramt_for_mcp.py](test_tool_foramt_for_mcp.py)** - Tests correct tool format for MCP
  - Validates LFM2's tool calling with nested format
  - Tests specific tool call markers
  - [Example Output](logs/test_tool_foramt_for_mcp_output.log)
  - [Documentation](test_tool_foramt_for_mcp_README.md)

- **[test_tool_format.py](test_tool_format.py)** - Deep diagnostic for various tool formats
  - Tests 8 different format hypotheses
  - Identifies which format is accepted by the server
  - [Example Output](logs/test_tool_format_output.log)
  - [Documentation](test_tool_format_README.md)

- **[test_toolcall.py](test_toolcall.py)** - Initial tool calling API test suite
  - Tests basic tool invocation and visibility
  - Includes multi-turn conversation tests
  - [Example Output](logs/test_toolcall_output.log)
  - [Documentation](test_toolcall_README.md)

### MCP Integration

- **[mcp_chain_discovery.py](mcp_chain_discovery.py)** - Chain depth discovery with real MCP operations
  - Tests actual chain depth limits with real tool calls
  - Discovers operational limits using MCP filesystem
  - [Example Output](logs/mcp_chain_discovery_output.log)
  - [Documentation](mcp_chain_discovery_README.md)

- **[mcp_chat.py](mcp_chat.py)** - Interactive chat with MCP filesystem tools
  - Provides real-time chat interface with filesystem tools
  - Includes anti-loop protection and conversation management
  - [Example Output](logs/mcp_chat_output.log)
  - [Documentation](mcp_chat_README.md)

- **[mcp_chat_enhanced.py](mcp_chat_enhanced.py)** - Enhanced tool transparency chat
  - Shows every step of tool execution process
  - Provides color-coded output for different stages
  - [Example Output](logs/mcp_chat_enhanced_output.log)
  - [Documentation](mcp_chat_enhanced_README.md)

- **[minimal_mcp.py](minimal_mcp.py)** - Minimal MCP example with directory operations
  - Demonstrates basic MCP integration
  - Tests directory listing, file creation, and reading operations
  - [Example Output](logs/minimal_mcp_output.log)
  - [Documentation](minimal_mcp_README.md)

### Research and Analysis

- **[deep_researcher.py](deep_researcher.py)** - Brave Search API integration for deep research
  - Performs multi-phase research using real APIs
  - Includes fact extraction, cross-validation, and synthesis
  - [Example Output](logs/deep_researcher_output.log)
  - [Documentation](deep_researcher_README.md)

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