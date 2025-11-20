# Test Tools Verbose

## Overview

Test Tools Verbose is an enhanced tool calling API test suite that provides full response logging to debug tool calling issues. It uses the correct nested tool format and includes detailed request/response logging to help identify server behavior and parsing issues.

## Key Features

- **Full Response Logging**: Shows complete server requests and responses
- **Correct Tool Format**: Uses the proper nested format (type + function object)
- **Four-Test Suite**: Tests basic calls, visibility, sequential calls, and multi-turn conversations
- **Format Variations**: Tests different tool response formats in multi-turn conversations
- **Detailed Timing Information**: Includes performance metrics from server responses
- **Comprehensive Tool Set**: Tests with 8 different function tools

## Architecture

### Core Components

- `chat()`: Makes API calls with verbose logging of requests and responses
- `extract_tool_calls()`: Parses tool calls from response content using special markers
- Complete server response logging with full JSON data
- Four different test scenarios with specific validation criteria

### Test Scenarios

#### Test 1: Basic Tool Call
- Tests a single tool call with get_weather
- Validates basic tool invocation capability
- Uses verbose logging of request/response

#### Test 2: Tool Visibility
- Tests if the model can "see" provided tools
- Checks how many tools can be recognized
- Tests up to 8 tools with mention detection

#### Test 3: Sequential Multi-Tool Calls
- Tests ability to call multiple tools in a single turn
- Validates up to 3 simultaneous tool calls
- Tests parsing of multiple tool calls in one response

#### Test 4: Multi-Turn Conversation
- Tests conversation continuity with tool results
- Validates tool result integration in conversation
- Tests follow-up queries based on tool results

#### Test 4B: Multi-Turn Format Variations
- Tests different tool response formats
- JSON array string format
- Plain string format
- Format with special response markers

## Configuration

- `BASE_URL`: Local LLM endpoint (default: "http://localhost:8080/v1")
- `MODEL_NAME`: Model to use for testing (default: "LFM2-8B-A1B-BF16-cuda")
- Includes 8 different test tools (weather, calculation, web search, time, translation, stock, email, task creation)
- Uses the correct nested tool format: `{"type": "function", "function": {...}}`

## Output

The test provides detailed output including:
- Complete request and response JSON
- Performance timing information
- Tool call extraction results
- Success/failure indicators for each test
- Multi-turn conversation state
- Summary of test results

## Results

Based on test execution:
- **Test 1 (Basic)**: ✓ PASS - Basic tool calling works correctly
- **Test 2 (Visibility)**: ✓ PASS - All 8 tools are visible to the model
- **Test 3 (Sequential)**: ✓ PASS - Up to 3 tools can be called in sequence
- **Test 4 (Multi-turn)**: ✗ FAIL - Model responds with natural language instead of tool calls after tool results
- **Test 4B**: Shows that JSON format and special markers work better for multi-turn tool calling

## Performance Metrics

The server provides detailed timing information:
- Prompt processing: ~20-25ms per token
- Prediction generation: ~32-33ms per token
- Total response time: varies by request complexity

## Limitations

- Multi-turn conversation tool calling fails in the first attempt due to model behavior
- Tool response format affects multi-turn success
- Depends on the availability of the LLM endpoint
- Some tool response formats work better than others

## Dependencies

- `requests` for API communication
- `json` for data serialization
- `typing` for type hints
- `re` for pattern matching