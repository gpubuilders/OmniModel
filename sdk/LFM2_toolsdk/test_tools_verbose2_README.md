# Test Tools Verbose2

## Overview

Test Tools Verbose2 is an advanced tool calling API test suite with corrected multi-turn tests. It addresses issues from previous tests by properly simulating tool execution, testing scenarios that actually require tools, and verifying that the model uses tool results in follow-up responses.

## Key Features

- **Proper Tool Execution Simulation**: Includes mock tool execution for realistic testing
- **Corrected Multi-Turn Tests**: Tests that verify the model's use of tool results
- **Context Verification**: Tests for model's ability to use previous results in new queries
- **Tool Response Format Testing**: Tests various response format approaches
- **Five Different Tools**: Includes weather, calculation, web search, stock price, and time tools
- **Comprehensive Validation**: Validates both tool calling and response behavior

## Architecture

### Core Components

- `chat()`: Makes API calls with detailed logging and response handling
- `extract_tool_calls()`: Parses tool calls from response content using special markers
- `execute_tool_call()`: Simulates actual tool execution with realistic mock data
- Four enhanced test scenarios with specific validation criteria

### Test Scenarios

#### Test 1: Basic Tool Call
- Tests a single tool call with get_weather
- Validates basic tool invocation capability
- Uses detailed logging of request/response

#### Test 2: Tool Visibility
- Tests if the model can "see" provided tools
- Checks how many tools can be recognized
- Tests all 5 available tools with mention detection

#### Test 3: Sequential Multi-Tool Calls
- Tests ability to call multiple tools in a single turn
- Validates up to 3 simultaneous tool calls
- Tests parsing of multiple tool calls in one response

#### Test 4A: Multi-Turn Basic
- Tests basic multi-turn conversation with tool execution
- Adds tool results to conversation history
- Validates follow-up tool calls

#### Test 4B: Multi-Turn Context Usage
- Tests use of previous tool results in new queries
- Verifies model can answer from context without unnecessary tools
- Tests comparison of results from different tool calls

#### Test 4C: Tool Response Format Variations
- Tests different tool response formats
- Validates special response markers
- Tests compatibility with chat template

## Mock Tool Execution

The test suite includes realistic mock implementations:
- **Weather**: Returns temperature, condition, humidity for specific cities
- **Stock Price**: Returns current price and change for stock symbols
- **Web Search**: Returns search results with count
- **Time**: Returns current time and date
- **Calculation**: Returns results for mathematical expressions

## Configuration

- `BASE_URL`: Local LLM endpoint (default: "http://localhost:8080/v1")
- `MODEL_NAME`: Model to use for testing (default: "LFM2-8B-A1B-BF16-cuda")
- Includes 5 different test tools (weather, calculation, web search, stock price, time)
- Uses the correct nested tool format: `{"type": "function", "function": {...}}`

## Output

The test provides detailed output including:
- Complete request and response information
- Performance metrics and token counts
- Tool call extraction results
- Success/failure indicators for each test
- Multi-turn conversation state
- Summary of test results

## Results

Based on test execution:
- **Test 1 (Basic)**: ✓ PASS - Basic tool calling works correctly
- **Test 2 (Visibility)**: ✓ PASS - All 5 tools are visible to the model
- **Test 3 (Sequential)**: ✓ PASS - Up to 3 tools can be called in sequence
- **Test 4A (Multi-turn Basic)**: ✓ PASS - Multi-turn tool execution works
- **Test 4B (Multi-turn Context)**: ✓ PASS - Model uses context properly
- **Test 4C (Format Variations)**: ✓ PASS - Different formats work correctly

## Performance Metrics

The server provides detailed timing information:
- Prompt processing and generation token rates
- Completion, prompt, and total token counts
- Response times for different request types

## Limitations

- Uses mock tool results instead of actual tool execution
- Depends on the availability of the LLM endpoint
- Tests may be affected by model-specific behavior

## Dependencies

- `requests` for API communication
- `json` for data serialization
- `typing` for type hints
- `re` for pattern matching