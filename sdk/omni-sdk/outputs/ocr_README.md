# OCR with Qwen3-Omni Local API

## Description
This script demonstrates Optical Character Recognition (OCR) using the Qwen3-Omni model through a local API endpoint. It extracts text from images containing both Chinese and English text, including mathematical formulas.

## Functionality
- **Local API Connection**: Connects to a local model server at `http://localhost:8080/v1/chat/completions`
- **Image Processing**: Loads and processes images using PIL and encodes them as base64
- **Text Extraction**: Extracts text from images in both Chinese and English
- **Mathematical Formula Recognition**: Can identify and extract mathematical formulas from images
- **Asset Handling**: Works with local image files stored in the `assets/` directory
- **Fallback Display**: Includes dummy functions for `IPython.display` if not running in a notebook environment

## Key Functions
- `run_model_local()`: Makes API calls to the local model server with messages and model parameters
- `process_ocr_message()`: Prepares messages for OCR tasks with images and text extraction prompts
- `get_local_file_path()`: Converts URLs to local file paths in the assets directory
- `load_local_image_as_base64()`: Loads local image files and encodes them as base64 strings

## Usage
```python
python ocr_local.py
```

## Configuration
- Model: Qwen3-Omni-10k
- API Endpoint: `http://localhost:8080/v1/chat/completions`
- Temperature: 0.1
- Max Tokens: 8192

## Requirements
- A running local LLM server with Qwen3-Omni-10k model accessible at `http://localhost:8080/v1`
- Image files in the `assets/` directory (ocr2.jpeg, ocr1.jpeg)
- Python libraries: `requests`, `numpy`, `PIL`, `base64`

## Examples
The script includes two OCR examples:
1. Chinese OCR (extracts Chinese text from an image)
2. English OCR (extracts mathematical formulas from an image)

## Notes
- Images are encoded as base64 and sent as data URLs in the format `data:image/jpeg;base64,...`
- If IPython is not available, dummy functions are used for displaying content
- Requires the local API to be running before execution