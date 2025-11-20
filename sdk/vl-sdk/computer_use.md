# Computer Use with Qwen3-VL (Local Version)

This example demonstrates how to use Qwen3-VL for computer use tasks. It takes a screenshot of a user's desktop and a query, then uses the model to interpret the user's query on the screenshot using a local LLM server instead of the DashScope API.

## Key Features

- **GUI Grounding**: Interprets user queries on desktop screenshots to identify specific UI elements
- **Coordinate Prediction**: Predicts specific coordinates (x, y) on the screen to interact with UI elements
- **Visual Annotation**: Draws points at the predicted coordinates to visualize where the model suggests to interact
- **Local API Integration**: Uses a local LLM server with OpenAI-compatible API endpoint

## Local Configuration

- **API Key**: `dummy-key` (local server may not validate)
- **Base URL**: `http://localhost:8080/v1`
- **Model ID**: `Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm`

## Examples Included

### Example 1: Open the Third Issue
Given a GitHub-style interface screenshot, the model is asked to "open the third issue" and identifies the specific coordinates of the third issue element.

### Example 2: Reload Cache
Given a screenshot of a page with a "Reload cache" button, the model identifies where to click to perform the action.

### Example 3: Open the First Issue
Given the same GitHub-style interface as Example 1, the model is asked to "open the first issue" and identifies the specific coordinates of the first issue element.

## Dependencies

This script automatically installs the following dependencies:
- Pillow
- requests
- openai
- qwen-agent

## Usage Requirements

- Local LLM server running at `http://localhost:8080`
- Required assets in `./assets/computer_use/` directory:
  - `computer_use1.jpeg`
  - `computer_use2.jpeg`

## Output

The script processes screenshots and provides:
- Text responses from the model explaining the action
- Visual annotations on the screenshot with green points at predicted coordinates
- Coordinate information in both relative (0-1000 scale) and absolute pixel values

## Coordinate System

The model uses a relative coordinate system from 0-1000 in both dimensions. The script converts these coordinates to actual pixel positions based on the image resizing parameters using smart image resizing with factors of 32.

## Image Processing

The script includes:
- Smart image resizing to ensure optimal processing dimensions
- Coordinate transformation from relative (0-1000) to absolute pixel coordinates
- Visual annotation with a green point at the predicted interaction location
- Center-radius highlighting for better visibility of target coordinates

## API Integration

Uses OpenAI-compatible client to communicate with the local server, handling image encoding to base64 format for transmission to the API server along with user queries.

## Error Handling

- Includes API connection testing functionality
- Provides informative error messages if assets are missing
- Handles coordinate parsing failures by defaulting to image center
- Handles both Jupyter and terminal environments appropriately
- Provides fallback visualization when coordinate parsing fails