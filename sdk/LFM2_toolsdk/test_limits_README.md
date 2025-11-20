# Test Limits

## Overview

Test Limits is a comprehensive tool testing framework that evaluates three critical axes of tool calling limitations in LLMs. It systematically tests parameter scaling, multiple tool selection, and sequential tool calls to determine the operational limits of the model's tool calling capabilities.

## Key Features

- **Three-Axis Testing**: Evaluates parameters, tools, and sequential calls separately
- **Parameter Scaling**: Tests single tool with increasing parameters (1-30)
- **Multiple Tools**: Tests tool selection from increasing tool sets (1-100)
- **Sequential Calls**: Tests sequential tool calls in a single turn (1-20)
- **Mock Execution**: Uses realistic mock tools to simulate responses
- **Complete Cycles**: Runs complete tool calling cycles with mock execution

## Architecture

### Core Components

- `call_api()`: Makes API calls to the LLM
- `extract_tool_calls()`: Parses tool calls from model responses
- `mock_tool_execution()`: Executes tools with generic mock responses
- `run_cycle()`: Runs complete tool calling cycles
- `generate_tool_with_n_params()`: Creates tools with specified number of parameters
- `generate_n_simple_tools()`: Creates specified number of simple tools

### Test Axes

#### Axis 1: Single Tool - Parameter Scaling
- Tests a single tool with increasing numbers of parameters
- Generates a tool with N parameters and tests with values
- Measures how many parameters are correctly included in calls
- Tests parameter counts: 1, 2, 5, 10, 15, 20, 25, 30

#### Axis 2: Multiple Tools - Simple Parameters
- Tests tool selection from increasingly large tool sets
- Tests ability to select the correct tool from N available tools
- Measures accuracy of tool selection across different tool counts
- Tests tool counts: 1, 5, 10, 20, 30, 50, 75, 100

#### Axis 3: Sequential Calls - Simple Tools
- Tests ability to make multiple sequential tool calls in one turn
- Tests calling multiple tools in a specified order
- Measures success rate of sequential tool execution
- Tests call counts: 1, 2, 3, 5, 7, 10, 15, 20

## Configuration

- `BASE_URL`: Local LLM endpoint (default: "http://localhost:8080/v1")
- `MODEL_NAME`: Model to use for testing (default: "LFM2-1.2B-Tool-Q4_K_M-cuda")

## Output

The test provides detailed output for each axis:
- Step-by-step testing with parameter/tool counts
- Success/failure indicators for each test point
- Parameter inclusion counts
- Tool selection accuracy
- Sequential call success rates
- Final summary with limits for all three axes

## Results

Based on test execution:
- **Axis 1 (Parameters)**: 30 max parameters (all 30 parameters successfully included)
- **Axis 2 (Multiple Tools)**: 100 max tools (correct tool selection maintained up to 100 tools)
- **Axis 3 (Sequential)**: 20 max sequential calls (successful sequential execution up to 20 calls)

## Limitations

- Uses mock tools instead of real services
- Tests with simplified parameters and tools
- Results may vary depending on model and parameters used
- Sequential calls test shows some inconsistency (failed at 10 but succeeded at 15)

## Dependencies

- `requests` for API communication
- `json` for data serialization
- `typing` for type hints
- `re` for pattern matching