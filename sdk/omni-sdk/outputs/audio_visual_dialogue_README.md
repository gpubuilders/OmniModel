# Audio Visual Dialogue with Qwen3-Omni Local API

## Description
This script demonstrates audio-visual dialogue using the Qwen3-Omni model through a local API endpoint. It handles conversations with audio and video inputs using system prompts to control the AI's response style and behavior.

## Functionality
- **Local API Connection**: Connects to a local model server at `http://localhost:8080/v1/chat/completions`
- **Audio Processing**: Loads and processes audio files using `librosa` and encodes them as base64
- **Video Processing**: Manages video files with text-based approach (since video may not be supported by all local models)
- **System Prompt Control**: Allows setting system prompts to control AI personality and dialogue behavior
- **Dialogue Management**: Handles both audio and video-based conversations with appropriate system instructions
- **Asset Handling**: Works with local audio and video files stored in the `assets/` directory
- **Fallback Display**: Includes dummy functions for `IPython.display` if not running in a notebook environment

## Key Functions
- `run_model_local()`: Makes API calls to the local model server with messages and model parameters
- `process_audio_dialogue_message()`: Prepares messages for audio-based dialogue with system prompt
- `process_video_dialogue_message()`: Prepares messages for video-based dialogue with system prompt (text-based approach)
- `get_local_file_path()`: Converts URLs to local file paths in the assets directory
- `load_local_audio_as_base64()`: Loads local audio files and encodes them as base64 strings
- `load_local_video_as_base64()`: Loads local video files and encodes them as base64 strings

## Usage
```python
python audio_visual_dialogue_local.py
```

## Configuration
- Model: Qwen3-Omni-10k
- API Endpoint: `http://localhost:8080/v1/chat/completions`
- Temperature: 0.1
- Max Tokens: 8192
- Audio Sample Rate: 16000 Hz

## Requirements
- A running local LLM server with Qwen3-Omni-10k model accessible at `http://localhost:8080/v1`
- Audio file in the `assets/` directory (translate_to_chinese.wav)
- Video file in the `assets/` directory (draw.mp4)
- Python libraries: `requests`, `numpy`, `PIL`, `librosa`, `base64`

## Examples
The script includes two dialogue examples:
1. Audio dialogue example (with detailed system prompt for virtual voice assistant behavior)
2. Video dialogue example (with system prompt for voice assistant characteristics)

## Notes
- Audio files are encoded as base64 and sent as data URLs in the format `data:audio/wav;base64,...`
- Video interactions currently use a text-based approach since video may not be supported by all local models
- The system prompts enforce specific dialogue rules like using appropriate pronouns and concise responses
- If IPython is not available, dummy functions are used for displaying content
- Requires the local API to be running before execution