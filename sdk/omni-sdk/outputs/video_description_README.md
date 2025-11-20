# Video Description with Qwen3-Omni Local API

## Description
This script demonstrates how to perform video description using the Qwen3-Omni model through a local API endpoint. It uses the `requests` library to connect to a locally running model server instead of calling remote transformers or vLLM directly.

## Functionality
- **Local API Connection**: Connects to a local model server at `http://localhost:8080/v1/chat/completions`
- **Video Processing**: Processes video description prompts using text-based approach (since video may not be supported by all local models)
- **Asset Handling**: Designed to work with local video files stored in the `assets/` directory
- **Fallback Display**: Includes dummy functions for `IPython.display` if not running in a notebook environment

## Key Functions
- `run_model_local()`: Makes API calls to the local model server with messages and model parameters
- `process_video_description_message()`: Prepares messages for video description tasks
- `get_local_file_path()`: Converts URLs to local file paths in the assets directory
- `load_local_video_as_base64()`: Loads local video files and encodes them as base64 strings

## Usage
```python
python video_description_local.py
```

## Configuration
- Model: Qwen3-Omni-10k
- API Endpoint: `http://localhost:8080/v1/chat/completions`
- Temperature: 0.1
- Max Tokens: 8192

## Requirements
- A running local LLM server with Qwen3-Omni-10k model accessible at `http://localhost:8080/v1`
- Video files in the `assets/` directory
- Python libraries: `requests`, `numpy`, `PIL`, `librosa`, `base64`

## Example Output
The script processes a video file (`assets/video1.mp4`) with a description prompt and returns the model's response based on the video content.

## Notes
- The script uses a text-based approach for video since video input may not be supported by all local models
- If IPython is not available, dummy functions are used for displaying content
- Requires the local API to be running before execution