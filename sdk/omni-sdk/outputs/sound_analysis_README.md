# Sound Analysis with Qwen3-Omni Local API

## Description
This script demonstrates sound analysis using the Qwen3-Omni model through a local API endpoint. It analyzes audio files to identify sounds, contexts, and locations based on the audio content.

## Functionality
- **Local API Connection**: Connects to a local model server at `http://localhost:8080/v1/chat/completions`
- **Audio Processing**: Loads and processes audio files using `librosa` and encodes them as base64
- **Sound Analysis**: Analyzes audio to identify what sounds are present and their context
- **Asset Handling**: Works with local audio files stored in the `assets/` directory
- **Fallback Display**: Includes dummy functions for `IPython.display` if not running in a notebook environment

## Key Functions
- `run_model_local()`: Makes API calls to the local model server with messages and model parameters
- `process_sound_analysis_message()`: Prepares messages for sound analysis tasks with audio and analysis prompts
- `get_local_file_path()`: Converts URLs to local file paths in the assets directory
- `load_local_audio_as_base64()`: Loads local audio files and encodes them as base64 strings

## Usage
```python
python sound_analysis_local.py
```

## Configuration
- Model: Qwen3-Omni-10k
- API Endpoint: `http://localhost:8080/v1/chat/completions`
- Temperature: 0.1
- Max Tokens: 8192
- Audio Sample Rate: 16000 Hz

## Requirements
- A running local LLM server with Qwen3-Omni-10k model accessible at `http://localhost:8080/v1`
- Audio files in the `assets/` directory (sound1.wav, sound2.mp3, sound3.mp3)
- Python libraries: `requests`, `numpy`, `PIL`, `librosa`, `base64`

## Examples
The script includes three sound analysis examples:
1. Analyzing what happened in audio (identifies dog barking and growling)
2. Identifying sound and context (identifies alarm clock and its typical usage)
3. Guessing location from sound (identifies restaurant kitchen based on cooking sounds)

## Notes
- Audio files are encoded as base64 and sent as data URLs in the format `data:audio/wav;base64,...`
- If IPython is not available, dummy functions are used for displaying content
- Requires the local API to be running before execution