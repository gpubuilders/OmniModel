# Image Math with Qwen3-Omni Local API

## Description
This script demonstrates image-based mathematical problem solving using the Qwen3-Omni model through a local API endpoint. It analyzes images containing mathematical problems and solves them using a combination of visual and text analysis.

## Functionality
- **Local API Connection**: Connects to a local model server at `http://localhost:8080/v1/chat/completions`
- **Image Processing**: Loads and processes mathematical problem images using PIL and encodes them as base64
- **Mathematical Problem Solving**: Solves complex mathematical problems that are presented in images
- **Multiple Choice Analysis**: Handles multiple choice questions with options for the model to select from
- **Asset Handling**: Works with local image files stored in the `assets/` directory
- **Fallback Display**: Includes dummy functions for `IPython.display` if not running in a notebook environment

## Key Functions
- `run_model_local()`: Makes API calls to the local model server with messages and model parameters
- `process_image_math_message()`: Prepares messages for image math problems with images, problem statements, and multiple choice options
- `get_local_file_path()`: Converts URLs to local file paths in the assets directory
- `load_local_image_as_base64()`: Loads local image files and encodes them as base64 strings

## Usage
```python
python image_math_local.py
```

## Configuration
- Model: Qwen3-Omni-10k
- API Endpoint: `http://localhost:8080/v1/chat/completions`
- Temperature: 0.1
- Max Tokens: 8192

## Requirements
- A running local LLM server with Qwen3-Omni-10k model accessible at `http://localhost:8080/v1`
- Image files in the `assets/` directory (5195.jpg, 4181.jpg)
- Python libraries: `requests`, `numpy`, `PIL`, `base64`

## Examples
The script includes two mathematical problem examples:
1. Lawn sprinkler mechanics problem (involving fluid mechanics and rotation rates)
2. Training set cost function problem (machine learning related mathematical problem)

## Notes
- Images are encoded as base64 and sent as data URLs in the format `data:image/jpeg;base64,...`
- If IPython is not available, dummy functions are used for displaying content
- Requires the local API to be running before execution
- The script can handle complex mathematical notation including LaTeX formatting