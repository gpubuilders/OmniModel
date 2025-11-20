# Cycles

## Overview

Cycles is a comprehensive tool calling test framework that follows clean architecture principles. It tests the complete cycle of tool calling: user query → tool call → tool execution → natural language response.

## Key Features

- **Complete Tool Cycle Testing**: Tests the entire flow from user query to final response
- **Mock Tool Execution**: Simulates tool responses without connecting to real services
- **Conversation Management**: Maintains conversation state across multiple turns
- **Multiple Test Scenarios**: Includes tests for single, multi-turn, and sequential tool calls
- **Result Validation**: Evaluates whether the model properly uses tool results

## Architecture

### Core Functions

- `call_api()`: Makes API calls to the LLM
- `extract_tool_calls()`: Parses tool calls from model responses
- `mock_tool_execution()`: Simulates tool results with predefined responses
- `run_tool_cycle()`: Executes a complete tool calling cycle

### Mock Tools

The framework includes mock implementations of these tools:
- `get_weather`: Returns temperature, condition, and humidity for a city
- `get_stock`: Returns stock price, change, and percentage for a symbol
- `search_web`: Returns search results for a query
- `calculate`: Returns a fixed result (42)

### Test Scenarios

The framework runs four main tests:
1. **Single Tool Cycle**: Tests a single tool call with complete cycle
2. **Multi-Turn Cycle**: Tests multi-turn conversations with different tools
3. **Sequential Tools**: Tests multiple tools in one turn
4. **Tool Result Usage**: Tests whether the model properly uses previous tool results

## Configuration

The framework uses the following configuration:
- `BASE_URL`: API endpoint (default: "http://localhost:8080/v1")
- `MODEL_NAME`: Model to use for testing (default: "LFM2-8B-A1B-BF16-cuda")
- `temperature`: API temperature setting (default: 0.3)

## Output Analysis

The test results show:
- **Single Tool Cycle**: ✓ PASS - Successfully calls weather tool and formats response
- **Multi-Turn Cycle**: ✓ PASS - Handles different tools across conversation turns
- **Sequential Tools**: ✓ PASS - Processes multiple tools in a single request
- **Tool Result Usage**: ✗ FAIL - Model fails to recall and compare previous tool results

Overall: 3/4 tests passed (75% success rate)

## Use Cases

This framework is useful for:
- Validating tool calling functionality
- Testing model's ability to process tool results
- Ensuring conversation context is maintained
- Debugging tool call parsing issues
- Testing sequential tool execution

## Limitations

- Uses mock tools instead of real services
- Model fails to recall previous tool results in complex scenarios
- Requires a running API endpoint to function
- Fixed mock responses that don't reflect real-world variability

## Dependencies

- `requests` for API communication
- `json` for data serialization
- `typing` for type hints
- `re` for pattern matching