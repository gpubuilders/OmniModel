# Mixed Audio Analysis with Qwen3-Omni Local API

## Description
This script demonstrates mixed audio analysis using the Qwen3-Omni model through a local API endpoint. It analyzes audio files to identify speaker characteristics (nationality, gender) and distinguish between sound effects and musical instruments present in the audio.

## Functionality
- **Local API Connection**: Connects to a local model server at `http://localhost:8080/v1/chat/completions`
- **Audio Processing**: Loads and processes audio files using `librosa` and encodes them as base64
- **Speaker Analysis**: Identifies speaker characteristics like nationality and gender
- **Sound Effect Recognition**: Detects and categorizes sound effects in the audio
- **Instrument Recognition**: Identifies musical instruments present in the audio
- **Asset Handling**: Works with local audio files stored in the `assets/` directory
- **Fallback Display**: Includes dummy functions for `IPython.display` if not running in a notebook environment

## Key Functions
- `run_model_local()`: Makes API calls to the local model server with messages and model parameters
- `process_mixed_audio_analysis_message()`: Prepares messages for mixed audio analysis tasks with audio and analysis prompts
- `get_local_file_path()`: Converts URLs to local file paths in the assets directory
- `load_local_audio_as_base64()`: Loads local audio files and encodes them as base64 strings

## Usage
```python
python mixed_audio_analysis_local.py
```

## Configuration
- Model: Qwen3-Omni-10k
- API Endpoint: `http://localhost:8080/v1/chat/completions`
- Temperature: 0.1
- Max Tokens: 8192
- Audio Sample Rate: 16000 Hz

## Requirements
- A running local LLM server with Qwen3-Omni-10k model accessible at `http://localhost:8080/v1`
- Audio files in the `assets/` directory (mixed_audio1.mp3, mixed_audio2.mp3)
- Python libraries: `requests`, `numpy`, `PIL`, `librosa`, `base64`

## Examples
The script includes two mixed audio analysis examples:
1. Nationality, gender and sound effects analysis (Chinese prompt: analyzes speaker nationality as French, gender as male, and identifies camera shutter sounds)
2. Sound effects and instruments analysis (English prompt: identifies sound effects and musical instruments separately in JSON format)

## Notes
- Audio files are encoded as base64 and sent as data URLs in the format `data:audio/mp3;base64,...`
- If IPython is not available, dummy functions are used for displaying content
- Requires the local API to be running before execution