# Audio Function Call with Qwen3-Omni Local API

## Description
This script demonstrates audio-based function calling using the Qwen3-Omni model through a local API endpoint. It processes audio input and determines which functions to call based on the audio content, with the model returning structured function call invocations.

## Functionality
- **Local API Connection**: Connects to a local model server at `http://localhost:8080/v1/chat/completions`
- **Audio Processing**: Loads and processes audio files using `librosa` and encodes them as base64
- **Function Calling**: Supports structured function calls based on audio input
- **Tool Definitions**: Provides predefined tools (web search and car AC control) for the model to use
- **Structured Response**: Returns function invocations in XML format with function names and arguments
- **Asset Handling**: Works with local audio files stored in the `assets/` directory
- **Fallback Display**: Includes dummy functions for `IPython.display` if not running in a notebook environment

## Key Functions
- `run_model_local()`: Makes API calls to the local model server with messages and model parameters
- `process_function_call_message()`: Prepares messages for audio function calling with system-defined tools
- `get_local_file_path()`: Converts URLs to local file paths in the assets directory
- `load_local_audio_as_base64()`: Loads local audio files and encodes them as base64 strings

## Usage
```python
python audio_function_call_local.py
```

## Configuration
- Model: Qwen3-Omni-10k
- API Endpoint: `http://localhost:8080/v1/chat/completions`
- Temperature: 0.1
- Max Tokens: 8192
- Audio Sample Rate: 16000 Hz

## Requirements
- A running local LLM server with Qwen3-Omni-10k model accessible at `http://localhost:8080/v1`
- Audio file in the `assets/` directory (functioncall_case.wav)
- Python libraries: `requests`, `numpy`, `PIL`, `librosa`, `base64`

## Available Functions
The script defines two available functions:
1. `web_search`: Uses web search engine to retrieve relevant information based on multiple queries
2. `car_ac_control`: Controls vehicle's air conditioning system to turn it on/off and set temperature

## Output Format
The model returns function calls within `<invoke></invoke>` XML tags, containing:
- Function name
- Arguments in JSON format

## Example
The example script processes an audio command that results in two function calls:
1. Turning on car AC at 22°C
2. Searching for malls near Alibaba's West Lake campus

## Notes
- Audio files are encoded as base64 and sent as data URLs in the format `data:audio/wav;base64,...`
- If IPython is not available, dummy functions are used for displaying content
- Requires the local API to be running before execution