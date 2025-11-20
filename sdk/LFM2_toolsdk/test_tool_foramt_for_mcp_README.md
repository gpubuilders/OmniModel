# Test Tool Format for MCP

## Overview

Test Tool Format for MCP is a verification tool that tests whether the LFM2 model correctly calls tools with the proper nested format. It specifically checks if the model recognizes and executes tools when provided with the correct OpenAI-style tool format and uses specific tool call markers.

## Key Features

- **Tool Format Verification**: Tests if LFM2 recognizes the correct nested tool format
- **Tool Call Detection**: Verifies that the model properly generates tool call markers
- **Multiple Test Cases**: Includes various query styles to test tool calling
- **Format Validation**: Shows the exact tool format being sent to the model
- **Correct Tool Identification**: Verifies that the right tool is called for each query

## Architecture

### Core Components

- Tool definitions with proper nested format (type → function → name, description, parameters)
- Three different test scenarios to verify tool calling
- Tool call detection using specific markers (`<|tool_call_start|>` and `<|tool_call_end|>`)
- Response parsing to extract and validate tool calls

### Tool Format

The test uses the correct nested format:
- `type`: "function"
- `function`: Contains name, description, and parameters
- `parameters`: Defines the object structure with properties and required fields

### Test Cases

#### Test 1: Weather Query
- Query: "What's the weather in Paris? Use the get_weather tool."
- Expected: get_weather tool with city parameter
- Verifies basic tool calling with natural language

#### Test 2: Math Query
- Query: "Calculate 15 + 27 using the calculate tool."
- Expected: calculate tool with expression parameter
- Tests mathematical operations

#### Test 3: Explicit Instruction
- Query: "YOU MUST CALL THE get_weather TOOL. Get weather for Tokyo."
- Expected: get_weather tool with city parameter
- Tests explicit tool calling instructions

## Configuration

- `LLM_URL`: Local LLM endpoint (default: "http://localhost:8080/v1")
- `MODEL`: Model to use for testing (default: "LFM2-1.2B-Tool-Q4_K_M-cuda")

## Output

The test provides detailed output for each test case:
- Query sent to the model
- Raw response showing tool call markers
- Extracted tool call content
- Success/failure indicators for tool detection
- Verification of correct tool selection
- Final verification of the tool format being sent

## Results

Based on test execution:
- All three test cases passed
- Model correctly generated tool call markers (`<|tool_call_start|>` and `<|tool_call_end|>`)
- Model correctly called the expected tools in each case
- Model properly formatted parameters in the tool calls
- LFM2 properly supports the nested tool format

## Limitations

- Tests only basic tool calling functionality
- Uses only two simple tools (weather and calculation)
- Doesn't test complex parameter structures
- Depends on the availability of the LLM endpoint

## Dependencies

- `requests` for API communication
- `json` for data serialization