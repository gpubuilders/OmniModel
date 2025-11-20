# Test Toolcall

## Overview

Test Toolcall is a comprehensive API test suite that evaluates a local LLM server's tool calling capabilities. It tests basic tool invocation, tool visibility, sequential multi-tool calls, and multi-turn conversations with tool results. The test suite reveals a critical issue with the server's tool parsing mechanism.

## Key Features

- **Four-Test Suite**: Tests basic calls, visibility, sequential calls, and multi-turn conversations
- **Tool Format Testing**: Verifies how the server handles tool definitions
- **Response Parsing**: Extracts and validates tool calls from responses
- **Debug Information**: Provides detailed API response data for troubleshooting
- **Structured Results**: Returns structured responses with success status and error details

## Architecture

### Core Components

- `chat()`: Makes API calls to the LLM with tools
- `extract_tool_calls()`: Parses tool calls from response content using special markers
- `APIResponse`: Data class for structured API response handling
- Four different test scenarios with specific validation criteria

### Test Scenarios

#### Test 1: Basic Tool Call
- Tests a single tool call with get_weather
- Validates basic tool invocation capability
- Currently fails due to server parsing issue

#### Test 2: Tool Visibility
- Tests if the model can "see" provided tools
- Checks how many tools can be recognized
- Determines maximum visible tools

#### Test 3: Sequential Multi-Tool Calls
- Tests ability to call multiple tools in a single turn
- Validates up to specified number of sequential calls
- Tests parsing of multiple tool calls

#### Test 4: Multi-Turn Conversation
- Tests conversation continuity with tool results
- Validates tool result integration in conversation
- Tests follow-up queries based on tool results

## Configuration

- `BASE_URL`: Local LLM endpoint (default: "http://localhost:8080/v1")
- `MODEL_NAME`: Model to use for testing (default: "LFM2-8B-A1B-BF16-cuda")
- Includes 8 different test tools (weather, calculation, web search, time, translation, stock, email, task creation)

## Output

The test provides detailed output including:
- API response debug information
- Tool call extraction results
- Success/failure indicators for each test
- Server error messages
- Summary of test results

## Results

Based on test execution:
- **Critical Failure**: The basic test fails with a server error: "Failed to parse tools: Missing tool function"
- The server expects a different tool format than what is provided
- The test uses the flat format: `{"type": "function", "name": "get_weather", ...}` but the server expects the nested format: `{"type": "function", "function": {...}}`
- All subsequent tests are not executed due to the critical failure

## Limitations

- The test suite fails at the first test due to server parsing issues
- Uses the incorrect tool format for this server
- Depends on the availability of the LLM endpoint
- All tests are skipped after the critical failure

## Dependencies

- `requests` for API communication
- `json` for data serialization
- `typing` for type hints
- `dataclasses` for structured response handling
- `re` for pattern matching