# Audio Visual Question with Qwen3-Omni Local API

## Description
This script demonstrates audio-visual question answering using the Qwen3-Omni model through a local API endpoint. It handles questions about video content, including both open-ended and multiple-choice questions.

## Functionality
- **Local API Connection**: Connects to a local model server at `http://localhost:8080/v1/chat/completions`
- **Video Processing**: Manages video files with text-based approach (since video may not be supported by all local models)
- **Question Answering**: Answers questions about video content and multiple-choice questions
- **Asset Handling**: Works with local video files stored in the `assets/` directory
- **Fallback Display**: Includes dummy functions for `IPython.display` if not running in a notebook environment

## Key Functions
- `run_model_local()`: Makes API calls to the local model server with messages and model parameters
- `process_video_question_message()`: Prepares messages for open-ended video questions with text-based approach
- `process_video_mcq_message()`: Prepares messages for multiple choice video questions with text-based approach
- `get_local_file_path()`: Converts URLs to local file paths in the assets directory
- `load_local_video_as_base64()`: Loads local video files and encodes them as base64 strings

## Usage
```python
python audio_visual_question_local.py
```

## Configuration
- Model: Qwen3-Omni-10k
- API Endpoint: `http://localhost:8080/v1/chat/completions`
- Temperature: 0.1
- Max Tokens: 8192

## Requirements
- A running local LLM server with Qwen3-Omni-10k model accessible at `http://localhost:8080/v1`
- Video files in the `assets/` directory (audio_visual.mp4, audio_visual2.mp4)
- Python libraries: `requests`, `numpy`, `PIL`, `librosa`, `base64`

## Examples
The script includes two video question examples:
1. Open-ended question about video content (asks for the first sentence a boy said when meeting a girl)
2. Multiple-choice question about video content (analyzes narrative purpose of question marks in a video)

## Notes
- The script currently uses a text-based approach for video content since video may not be supported by all local models
- If IPython is not available, dummy functions are used for displaying content
- Requires the local API to be running before execution