# Document Parsing with Qwen3-VL (Local Version)

This example showcases the powerful document parsing capabilities of Qwen3-VL. It can process any image and output its contents in various formats such as HTML, JSON, Markdown, and LaTeX. Notably, it introduces two unique Qwenvl formats:

- Qwenvl HTML format, which adds positional information for each component to enable precise document reconstruction and manipulation.
- Qwenvl Markdown format, which converts the overall image content into Markdown. In this format, all tables are represented in LaTeX with their corresponding coordinates indicated before each table, and images are replaced with coordinate-based placeholders for accurate positioning.

## Key Features

- **Multi-Format Output**: Supports various document formats including HTML, JSON, Markdown, and LaTeX
- **Qwenvl-Specific Formats**: Advanced formats with positional information for precise document reconstruction
- **Coordinate Visualization**: Visualizes bounding boxes on images based on parsed coordinates
- **Local API Integration**: Uses a local LLM server with OpenAI-compatible API endpoint

## Local Configuration

- **API Key**: `dummy-key` (local server may not validate)
- **Base URL**: `http://localhost:8080/v1`
- **Model ID**: `Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm`

## Examples Included

### Example 1: Document Parsing in QwenVL HTML Format
Converts an image document to HTML format with bounding box coordinates. The response includes data-bbox attributes that indicate the position of each document element. The script visualizes these bounding boxes on the image and provides a cleaned HTML output.

### Example 2: Document Parsing in QwenVL Markdown Format
Converts an image document to the specialized QwenVL Markdown format. In this format, tables are represented in LaTeX with coordinate information, and images are replaced with placeholders. The script visualizes these coordinate-based elements on the image.

## Dependencies

This script automatically installs the following dependencies:
- Pillow
- requests
- openai
- beautifulsoup4
- qwen-agent

## Usage Requirements

- Local LLM server running at `http://localhost:8080`
- Internet access to download remote images for processing (images are saved temporarily and cleaned up after processing)

## Output

The script processes document images and provides:
- Structured text representations of document content in various formats
- Visual annotations showing bounding boxes for document elements
- Coordinate information in 0-1000 relative scale for precise positioning
- Cleaned HTML without unnecessary styling information

## Visualization

The script includes specialized visualization functions:
- `draw_bbox_html`: Visualizes HTML elements with data-bbox attributes, drawing red rectangles around document components
- `draw_bbox_markdown`: Visualizes Markdown elements with coordinate comments, using blue boxes for images and red boxes for tables

## Format Processing

The script handles different document formats with specific processing:
- HTML format includes positional data-bbox attributes for precise element positioning
- Markdown format uses coordinate comments to indicate where tables and images should be placed
- Cleaned HTML output removes styling and unnecessary attributes while preserving structure

## API Integration

Uses OpenAI-compatible client to communicate with the local server, handling image encoding to base64 format for transmission to the API server. Includes configurable pixel limits for optimal image processing.

## Error Handling

- Includes API connection testing functionality
- Provides informative error messages
- Handles both Jupyter and terminal environments appropriately
- Automatically cleans up temporary image files after processing
- Handles coordinate parsing failures gracefully