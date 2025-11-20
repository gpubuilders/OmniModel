# Omni Recognition with Qwen3-VL (Local Version)

This example demonstrates Qwen3-VL's omni recognition capabilities using a local LLM server. It includes object recognition (celebrities, anime characters, landmarks, animals, plants), object spotting with bounding boxes, and multi-object identification.

## Key Features

- **Object Recognition**: Identifies celebrities, anime characters, landmarks, animals, plants, and other recognizable objects
- **Object Spotting**: Detects multiple objects in an image with precise bounding box coordinates
- **Multi-Language Support**: Provides names in both Chinese and English when applicable
- **Multi-Object Identification**: Comprehensive scene analysis identifying multiple objects simultaneously
- **Local API Integration**: Uses a local LLM server with direct HTTP requests to the API endpoint

## Local Configuration

- **Base URL**: `http://localhost:8080/v1`
- **Model ID**: `Qwen3-Omni-10k`

## Examples Included

### Example 1: Object Recognition
Recognizes individual objects like celebrities, anime characters, landmarks, animals, plants, etc. with a query like "Who is this?" or "What kind of bird is this?".

### Example 2: Object Spotting with Bounding Boxes
Identifies multiple objects in an image and provides their bounding box coordinates in JSON format along with names in both Chinese and English.

### Example 3: Multi-object Recognition
Performs comprehensive scene analysis to identify all recognizable objects in a complex image with descriptions.

## Dependencies

This script automatically installs the following dependencies:
- requests
- pillow

## Usage Requirements

- Local LLM server running at `http://localhost:8080`
- Required assets in `./assets/omni_recognition/` directory:
  - `sample-celebrity.jpeg` (for celebrity recognition)
  - `sample-celebrity-2.jpg` (for celebrity recognition)
  - `sample-anime.jpeg` (for anime character recognition)
  - `sample-food.jpeg` (for food recognition with bounding boxes)
  - `sample-bird.jpg` (for animal recognition)

## Output

The script processes images and provides:
- Identification of recognizable objects with detailed information
- Bounding box coordinates for detected objects in relative (0-1000) format
- Names in both Chinese and English for multilingual support
- Confidence scores when available
- Structured JSON output for object spotting results

## Recognition Capabilities

The omni recognition functionality includes:
- Celebrity identification with biographical information
- Anime and manga character recognition
- Food and dish identification
- Animal species identification
- Plant species identification
- Landmark and building recognition
- Product and brand recognition
- Vehicle identification

## Coordinate System

Object spotting uses a 0-1000 relative coordinate system. The script converts these relative coordinates to absolute pixel positions based on the image dimensions for proper display and understanding.

## API Integration

Uses direct HTTP requests to communicate with the local server, handling image encoding to base64 format for transmission to the API server. Includes configurable temperature and top_p parameters for generation control.

## Error Handling

- Includes server connection testing functionality
- Provides informative error messages if assets are missing
- Handles JSON parsing failures for structured outputs
- Reports status codes for API communication issues
- Validates image file existence before processing

## Object Spotting Format

Object spotting results are returned in JSON format:
```json
[
  {
    "name_en": "Rice Bowl",
    "name_cn": "米饭",
    "bbox": [100, 200, 300, 400]
  }
]
```
Coordinates are in relative format (0-1000) and are converted to absolute pixel coordinates for display.