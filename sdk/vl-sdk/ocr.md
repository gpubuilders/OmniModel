# OCR with Qwen3-VL (Local Version)

This example demonstrates Qwen3-VL's OCR capabilities using a local LLM server. It includes full-page OCR for English and multilingual text, text spotting with bounding boxes, and visual information extraction from documents.

## Key Features

- **Full-Page OCR**: Extracts all text from images in both English and multilingual contexts
- **Text Spotting**: Identifies text elements with precise bounding box coordinates
- **Information Extraction**: Extracts specific key-value pairs from structured documents like receipts and invoices
- **Local API Integration**: Uses a local LLM server with direct HTTP requests to the API endpoint

## Local Configuration

- **Base URL**: `http://localhost:8080/v1`
- **Model ID**: `Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm`

## Examples Included

### Example 1: Full-page OCR for English Text
Reads all text content from an image with a simple prompt "Read all the text in the image."

### Example 2: Full-page OCR for Multilingual Text
Extracts text content from images containing multilingual text with a prompt that requests only text content without additional formatting.

### Example 3: Text Spotting (Line-level)
Detects text in line-level granularity and outputs bounding box coordinates in JSON format along with the text content.

### Example 4: Text Spotting (Word-level)
Detects text in word-level granularity and outputs bounding box coordinates in JSON format along with the text content.

### Example 5: Visual Information Extraction
Extracts specific key-value information from structured documents (like receipts or invoices) based on specified keys.

## Dependencies

This script automatically installs the following dependencies:
- requests
- pillow

## Usage Requirements

- Local LLM server running at `http://localhost:8080`
- Required assets in `./assets/ocr/` directory:
  - `ocr_example2.jpg` (for English OCR and word-level spotting)
  - `ocr_example3.jpg` (for text spotting and information extraction)
  - `ocr_example6.jpg` (for multilingual OCR)

## Output

The script processes images and provides:
- Extracted text content from images
- Bounding box coordinates for text elements in relative (0-999) format
- Structured JSON output for text spotting results
- Key-value pairs for information extraction tasks

## Coordinate System

Text spotting uses a 0-999 relative coordinate system. The script converts these relative coordinates to absolute pixel positions based on the image dimensions for proper display and understanding.

## OCR Capabilities

The OCR functionality includes:
- Line-level text detection with bounding boxes
- Word-level text detection with bounding boxes
- Multilingual text recognition
- Structured document information extraction
- Configurable token limits for different output lengths

## API Integration

Uses direct HTTP requests to communicate with the local server, handling image encoding to base64 format for transmission to the API server. Includes configurable temperature and top_p parameters for generation control.

## Error Handling

- Includes server connection testing functionality
- Provides informative error messages if assets are missing
- Handles JSON parsing failures for structured outputs
- Reports status codes for API communication issues
- Validates image file existence before processing

## Text Spotting Format

Text spotting results are returned in JSON format:
```json
[
  {
    "bbox_2d": [x1, y1, x2, y2], 
    "text_content": "text"
  }
]
```
Coordinates are in relative format (0-999) and are converted to absolute pixel coordinates for display.