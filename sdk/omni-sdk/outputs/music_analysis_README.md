# Music Analysis with Qwen3-Omni Local API

## Description
This script demonstrates music analysis using the Qwen3-Omni model through a local API endpoint. It analyzes music files to identify styles, instruments, rhythms, dynamics, and emotional content.

## Functionality
- **Local API Connection**: Connects to a local model server at `http://localhost:8080/v1/chat/completions`
- **Audio Processing**: Loads and processes music files using `librosa` and encodes them as base64
- **Music Analysis**: Analyzes music to identify style, genre, instruments, rhythm, and emotional content
- **Multilingual Support**: Supports music analysis with both Chinese and English prompts
- **Asset Handling**: Works with local music files stored in the `assets/` directory (including files with Chinese names)
- **Fallback Display**: Includes dummy functions for `IPython.display` if not running in a notebook environment

## Key Functions
- `run_model_local()`: Makes API calls to the local model server with messages and model parameters
- `process_music_analysis_message()`: Prepares messages for music analysis tasks with audio and analysis prompts
- `get_local_file_path()`: Converts URLs to local file paths in the assets directory
- `load_local_audio_as_base64()`: Loads local audio files and encodes them as base64 strings

## Usage
```python
python music_analysis_local.py
```

## Configuration
- Model: Qwen3-Omni-10k
- API Endpoint: `http://localhost:8080/v1/chat/completions`
- Temperature: 0.1
- Max Tokens: 8192
- Audio Sample Rate: 16000 Hz

## Requirements
- A running local LLM server with Qwen3-Omni-10k model accessible at `http://localhost:8080/v1`
- Music files in the `assets/` directory (音乐风格-调性.mp3, 37573.mp3, 353349.mp3)
- Python libraries: `requests`, `numpy`, `PIL`, `librosa`, `base64`

## Examples
The script includes three music analysis examples:
1. Music style analysis (Chinese prompt: "请分析这是什么风格的音乐？")
2. Detailed music analysis (English prompt: analyzes style, rhythm, dynamics, emotions, instruments)
3. Appreciative music analysis (analyzes style, genre, and instrumental collaboration patterns)

## Notes
- Audio files are encoded as base64 and sent as data URLs in the format `data:audio/mp3;base64,...`
- If IPython is not available, dummy functions are used for displaying content
- Requires the local API to be running before execution
- Handles files with non-ASCII characters in their names (Chinese filenames)