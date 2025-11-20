# Image Question with Qwen3-Omni Local API

## Description
This script demonstrates image-based question answering using the Qwen3-Omni model through a local API endpoint. It analyzes images and responds to various types of questions about the visual content, including style analysis, prediction, and pattern recognition.

## Functionality
- **Local API Connection**: Connects to a local model server at `http://localhost:8080/v1/chat/completions`
- **Image Processing**: Loads and processes images using PIL and encodes them as base64
- **Multimodal Analysis**: Combines visual and text inputs for comprehensive image analysis
- **Question Answering**: Answers various types of questions about images (style, prediction, pattern recognition)
- **Asset Handling**: Works with local image files stored in the `assets/` directory
- **Fallback Display**: Includes dummy functions for `IPython.display` if not running in a notebook environment

## Key Functions
- `run_model_local()`: Makes API calls to the local model server with messages and model parameters
- `process_multimodal_message()`: Prepares multimodal messages combining images and text prompts
- `get_local_file_path()`: Converts URLs to local file paths in the assets directory
- `load_local_image_as_base64()`: Loads local image files and encodes them as base64 strings

## Usage
```python
python image_question_local.py
```

## Configuration
- Model: Qwen3-Omni-10k
- API Endpoint: `http://localhost:8080/v1/chat/completions`
- Temperature: 0.1
- Max Tokens: 8192

## Requirements
- A running local LLM server with Qwen3-Omni-10k model accessible at `http://localhost:8080/v1`
- Image files in the `assets/` directory (2621.jpg, 2233.jpg, val_IQ_Test_113.jpg)
- Python libraries: `requests`, `numpy`, `PIL`, `base64`

## Examples
The script includes three different image analysis examples:
1. Art style identification (identifies a Salvador Dali painting as Surrealist)
2. Predictive scene analysis (predicts stormy weather based on cloud formations)
3. Pattern recognition (answers an IQ test question by identifying the correct option)

## Notes
- Images are encoded as base64 and sent as data URLs in the format `data:image/jpeg;base64,...`
- If IPython is not available, dummy functions are used for displaying content
- Requires the local API to be running before execution