# Audio Caption with Qwen3-Omni Local API

## Description
This script demonstrates audio captioning using the Qwen3-Omni model through a local API endpoint. It analyzes audio files and provides detailed descriptions of the audio content, including sound types, environmental characteristics, and contextual information.

## Functionality
- **Local API Connection**: Connects to a local model server at `http://localhost:8080/v1/chat/completions`
- **Audio Processing**: Loads and processes audio files using `librosa` and encodes them as base64
- **Audio Analysis**: Provides detailed descriptions of audio content, including sound types, environment, and context
- **Multimodal Processing**: Combines audio and text inputs for comprehensive analysis
- **Asset Handling**: Works with local audio files stored in the `assets/` directory
- **Fallback Display**: Includes dummy functions for `IPython.display` if not running in a notebook environment

## Key Functions
- `run_model_local()`: Makes API calls to the local model server with messages and model parameters
- `process_multimodal_message()`: Prepares multimodal messages combining audio and text prompts
- `get_local_file_path()`: Converts URLs to local file paths in the assets directory
- `load_local_audio_as_base64()`: Loads local audio files and encodes them as base64 strings

## Usage
```python
python audio_caption_local.py
```

## Configuration
- Model: Qwen3-Omni-10k
- API Endpoint: `http://localhost:8080/v1/chat/completions`
- Temperature: 0.1
- Max Tokens: 8192
- Audio Sample Rate: 16000 Hz

## Requirements
- A running local LLM server with Qwen3-Omni-10k model accessible at `http://localhost:8080/v1`
- Audio files in the `assets/` directory (caption1.mp3, caption2.mp3, caption3.mp3)
- Python libraries: `requests`, `numpy`, `PIL`, `librosa`, `base64`

## Examples
The script includes three audio captioning examples:
1. Industrial environment (describes a factory/warehouse with mechanical sounds)
2. Lo-fi acoustic pop song (describes a home recording with vocals and guitar)
3. Intimate reflection with background music (describes a personal recording with spoken words over music)

## Notes
- Audio files are encoded as base64 and sent as data URLs in the format `data:audio/mp3;base64,...`
- If IPython is not available, dummy functions are used for displaying content
- Requires the local API to be running before execution