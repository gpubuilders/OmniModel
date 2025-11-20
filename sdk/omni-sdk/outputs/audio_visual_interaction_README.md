# Audio Visual Interaction with Qwen3-Omni Local API

## Description
This script demonstrates audio-visual interaction using the Qwen3-Omni model through a local API endpoint. It handles various interaction modes with different system prompts to control the AI's behavior, including audio-only interactions and system-assisted audio/video interactions.

## Functionality
- **Local API Connection**: Connects to a local model server at `http://localhost:8080/v1/chat/completions`
- **Audio Processing**: Loads and processes audio files using `librosa` and encodes them as base64
- **Video Processing**: Manages video files with text-based approach (since video may not be supported by all local models)
- **System Prompt Control**: Allows setting system prompts to control AI personality and behavior
- **Multiple Interaction Modes**: Supports audio-only, system+audio, and system+video interaction modes
- **Asset Handling**: Works with local audio and video files stored in the `assets/` directory
- **Fallback Display**: Includes dummy functions for `IPython.display` if not running in a notebook environment

## Key Functions
- `run_model_local()`: Makes API calls to the local model server with messages and model parameters
- `process_audio_interaction_message()`: Prepares messages for audio-only interactions
- `process_system_audio_interaction_message()`: Prepares messages for system + audio interactions
- `process_system_video_interaction_message()`: Prepares messages for system + video interactions (text-based approach)
- `get_local_file_path()`: Converts URLs to local file paths in the assets directory
- `load_local_audio_as_base64()`: Loads local audio files and encodes them as base64 strings

## Usage
```python
python audio_visual_interaction_local.py
```

## Configuration
- Model: Qwen3-Omni-10k
- API Endpoint: `http://localhost:8080/v1/chat/completions`
- Temperature: 0.1
- Max Tokens: 8192
- Audio Sample Rate: 16000 Hz

## Requirements
- A running local LLM server with Qwen3-Omni-10k model accessible at `http://localhost:8080/v1`
- Audio files in the `assets/` directory (interaction1.mp3, interaction3.mp3)
- Video files in the `assets/` directory (interaction2.mp4, interaction4.mp4)
- Python libraries: `requests`, `numpy`, `PIL`, `librosa`, `base64`

## Examples
The script includes three interaction examples:
1. Audio-only interaction (provides swimming tips based on audio input)
2. System + audio interaction with romantic/artistic AI personality (responds in Chinese with metaphorical language)
3. System + video interaction with Beijing "大爷" personality (responds in Chinese as a humorous Beijing elder)

## Notes
- Audio files are encoded as base64 and sent as data URLs in the format `data:audio/mp3;base64,...`
- Video interactions currently use a text-based approach since video may not be supported by all local models
- If IPython is not available, dummy functions are used for displaying content
- Requires the local API to be running before execution