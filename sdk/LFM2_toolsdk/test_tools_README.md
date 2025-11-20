# Test Tools

## Overview

Test Tools is a working tool calling API test suite that uses the correct OpenAI nested format for tools. It provides a streamlined test of the basic tool calling capabilities with the proper format that works with the LFM2 server.

## Key Features

- **Correct Tool Format**: Uses the proper OpenAI nested format with type and function object
- **Simple Test Suite**: Tests basic tool calling, visibility, sequential calls, and multi-turn conversations
- **Eight Different Tools**: Includes weather, math, web search, time, translation, stock, email, and task tools
- **Clean Implementation**: Minimal error handling and straightforward test implementation
- **Tool Call Extraction**: Parses tool calls from responses using special markers

## Architecture

### Core Components

- `chat()`: Makes API calls to the LLM with tools
- `extract_tool_calls()`: Parses tool calls from response content using special markers
- Four different test scenarios with specific validation criteria

### Tool Format

The test uses the correct nested format:
```
{
  "type": "function",
  "function": {
    "name": "...",
    "description": "...",
    "parameters": {...}
  }
}
```

### Test Scenarios

#### Test 1: Basic Tool Call
- Tests a single tool call with get_weather
- Validates basic tool invocation capability
- Extracts and verifies tool calls

#### Test 2: Tool Visibility
- Tests if the model can "see" provided tools
- Checks how many tools can be recognized
- Tests all 8 available tools with mention detection

#### Test 3: Sequential Multi-Tool Calls
- Tests ability to call multiple tools in a single turn
- Validates up to 3 simultaneous tool calls
- Tests parsing of multiple tool calls in one response

#### Test 4: Multi-Turn Conversation
- Tests conversation continuity with tool results
- Validates tool result integration in conversation
- Tests follow-up queries based on tool results

## Configuration

- `BASE_URL`: Local LLM endpoint (default: "http://localhost:8080/v1")
- `MODEL_NAME`: Model to use for testing (default: "LFM2-8B-A1B-BF16-cuda")
- Includes 8 different test tools (weather, calculation, web search, time, translation, stock, email, task creation)
- Uses the correct nested tool format: `{"type": "function", "function": {...}}`

## Output

The test provides concise output for each test:
- Tool call extraction results
- Success/failure indicators for each test
- Summary of test results

## Results

Based on test execution:
- **Test 1 (Basic)**: ✓ PASS - Basic tool calling works correctly
- **Test 2 (Visibility)**: ✓ PASS - All 8 tools are visible to the model
- **Test 3 (Sequential)**: ✓ PASS - Up to 3 tools can be called in sequence
- **Test 4 (Multi-turn)**: ✗ FAIL - Model doesn't make tool calls in follow-up turn after tool result

## Limitations

- Multi-turn conversation test fails (model responds with natural language instead of tool calls)
- Uses minimal error handling
- Depends on the availability of the LLM endpoint
- Simple mock tool response in multi-turn test

## Dependencies

- `requests` for API communication
- `json` for data serialization
- `typing` for type hints
- `re` for pattern matching