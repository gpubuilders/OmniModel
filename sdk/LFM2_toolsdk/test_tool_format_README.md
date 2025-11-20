# Test Tool Format

## Overview

Test Tool Format is a deep diagnostic tool that systematically tests various tool format structures to determine which one is accepted by the LFM2 server. It addresses the issue where the server gives contradictory errors when different tool formats are used, suggesting complex parsing requirements.

## Key Features

- **Systematic Testing**: Tests 8 different tool format hypotheses
- **Diagnostic Approach**: Investigates why certain formats fail with specific error messages
- **Format Validation**: Verifies which tool structure is properly parsed by the server
- **Error Analysis**: Provides detailed error messages to understand server requirements
- **OpenAI Format Verification**: Tests the standard OpenAI nested format (which succeeds)

## Architecture

### Core Components

- `test_format()`: Tests a specific tool format with detailed error handling
- Multiple format hypotheses testing different tool structure approaches
- Error message parsing to extract relevant information
- Response validation to check for successful tool calls

### Test Formats

The script tests these 8 different tool format hypotheses:

#### Format 1: OpenAI nested structure (type + function object)
- Correct structure: `{"type": "function", "function": {...}}`
- This is the standard OpenAI format and the one that works

#### Format 2: type='tool' with function object
- Testing if server accepts "tool" as the type value

#### Format 3: Flat with type='function' and function=name
- Testing a different approach where function is the name

#### Format 4: type='callable'
- Testing if server supports "callable" as a type

#### Format 5: type='string'
- Testing a potentially non-standard type value

#### Format 6: function as top-level wrapper
- Testing if function should be the top-level field

#### Format 7: Full JSON schema format
- Testing with full JSON schema compliance

#### Format 8: Direct properties (no parameters wrapper)
- Testing without the parameters wrapper object

## Configuration

- `BASE_URL`: Local LLM endpoint (default: "http://localhost:8080/v1")
- `MODEL_NAME`: Model to use for testing (default: "LFM2-8B-A1B-BF16-cuda")

## Output

The test provides detailed output for each format:
- Tool structure being tested
- Success/failure indicators
- Error messages when format is rejected
- Response content when successful
- Detected tool calls in successful responses

## Results

Based on test execution:
- Format 1 (OpenAI nested structure) succeeds
- All other formats fail with various error messages
- Format 1 generates proper tool call markers (`<|tool_call_start|>` and `<|tool_call_end|>`)
- The server expects the nested format: `{"type": "function", "function": {...}}`

## Error Categories

The server produces these types of errors:
- "Unsupported tool type": When the type field has an unsupported value
- "Missing tool function/type": When required fields are missing
- JSON parsing errors: When structure doesn't match expected schema

## Limitations

- Tests only 8 specific format hypotheses
- Uses only one simple tool (get_weather) for all tests
- Depends on the availability of the LLM endpoint
- Results may vary depending on server configuration

## Dependencies

- `requests` for API communication
- `json` for data serialization
- `typing` for type hints