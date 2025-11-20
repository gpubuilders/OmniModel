# Speech Translation with Qwen3-Omni Local API

## Description
This script demonstrates speech translation using the Qwen3-Omni model through a local API endpoint. It allows for translating speech from one language to another (e.g., Chinese to English, English to Chinese, French to English) using audio files as input.

## Functionality
- **Local API Connection**: Connects to a local model server at `http://localhost:8080/v1/chat/completions`
- **Audio Processing**: Loads and processes audio files using `librosa` and encodes them as base64
- **Multilingual Support**: Supports translation between multiple languages (Chinese, English, French)
- **Asset Handling**: Works with local audio files stored in the `assets/` directory
- **Fallback Display**: Includes dummy functions for `IPython.display` if not running in a notebook environment

## Key Functions
- `run_model_local()`: Makes API calls to the local model server with messages and model parameters
- `process_speech_translation_message()`: Prepares messages for speech translation tasks with audio and translation prompts
- `get_local_file_path()`: Converts URLs to local file paths in the assets directory
- `load_local_audio_as_base64()`: Loads local audio files and encodes them as base64 strings

## Usage
```python
python speech_translation_local.py
```

## Configuration
- Model: Qwen3-Omni-10k
- API Endpoint: `http://localhost:8080/v1/chat/completions`
- Temperature: 0.1
- Max Tokens: 8192
- Audio Sample Rate: 16000 Hz

## Requirements
- A running local LLM server with Qwen3-Omni-10k model accessible at `http://localhost:8080/v1`
- Audio files in the `assets/` directory (asr_zh.wav, asr_en.wav, asr_fr.wav)
- Python libraries: `requests`, `numpy`, `PIL`, `librosa`, `base64`

## Examples
The script includes three translation examples:
1. Chinese to English translation
2. English to Chinese translation  
3. French to English translation

## Notes
- Audio files are encoded as base64 and sent as data URLs in the format `data:audio/wav;base64,...`
- If IPython is not available, dummy functions are used for displaying content
- Requires the local API to be running before execution