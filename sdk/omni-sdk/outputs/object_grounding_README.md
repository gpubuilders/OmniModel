# Object Grounding with Qwen3-Omni Local API

## Description
This script demonstrates object grounding using the Qwen3-Omni model through a local API endpoint. It identifies and locates specific objects within images by returning bounding box coordinates and labels.

## Functionality
- **Local API Connection**: Connects to a local model server at `http://localhost:8080/v1/chat/completions`
- **Image Processing**: Loads and processes images using PIL and encodes them as base64
- **Object Detection**: Locates specific objects within images and returns bounding box coordinates
- **Multi-Object Recognition**: Can identify multiple instances of the same object in an image
- **Asset Handling**: Works with local image files stored in the `assets/` directory
- **Fallback Display**: Includes dummy functions for `IPython.display` if not running in a notebook environment

## Key Functions
- `run_model_local()`: Makes API calls to the local model server with messages and model parameters
- `process_object_grounding_message()`: Prepares messages for object grounding tasks with images and object descriptions
- `get_local_file_path()`: Converts URLs to local file paths in the assets directory
- `load_local_image_as_base64()`: Loads local image files and encodes them as base64 strings

## Usage
```python
python object_grounding_local.py
```

## Configuration
- Model: Qwen3-Omni-10k
- API Endpoint: `http://localhost:8080/v1/chat/completions`
- Temperature: 0.1
- Max Tokens: 8192

## Requirements
- A running local LLM server with Qwen3-Omni-10k model accessible at `http://localhost:8080/v1`
- Image files in the `assets/` directory (grounding1.jpeg, grounding2.jpg)
- Python libraries: `requests`, `numpy`, `PIL`, `base64`

## Examples
The script includes two object grounding examples:
1. Locate bird (identifies and returns bounding boxes for multiple birds in an image)
2. Locate person riding motorcycle (identifies and returns bounding box for a person riding a motorcycle with helmet)

## Output Format
The model returns JSON-formatted response containing an array of objects with:
- `bbox_2d`: An array of four coordinates [x1, y1, x2, y2] representing the bounding box
- `label`: The label of the detected object

## Notes
- Images are encoded as base64 and sent as data URLs in the format `data:image/jpeg;base64,...`
- If IPython is not available, dummy functions are used for displaying content
- Requires the local API to be running before execution