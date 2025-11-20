# Think with Images with Qwen3-VL (Local Version)

This example demonstrates how to use Qwen3-VL with image zoom-in and search capabilities using a local LLM server instead of external APIs. It showcases advanced visual analysis with iterative reasoning and tool usage.

## Key Features

- **Image Zoom-in Capability**: Allows detailed analysis of specific regions within images
- **Multi-functional Tools**: Supports both zoom-in and image search capabilities (when API keys available)
- **Iterative Visual Analysis**: Implements a structured thinking process for image analysis
- **Local API Integration**: Uses a local LLM server with Qwen-Agent framework

## Local Configuration

- **Base URL**: `http://localhost:8080/v1`
- **Model ID**: `Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm`
- **API Key**: `EMPTY` (for local server)

## Examples Included

### Example 1: Zoom-in Assistant
Creates an agent that performs detailed visual analysis with zoom-in capabilities. The agent follows a structured thinking process:
- First, looks closely at the image and provides detailed descriptions
- Uses the image_zoom_in_tool to examine specific regions in detail
- Continues analysis until the question is fully answered

### Example 2: Multi-functional Assistant (Zoom + Search)
Creates an agent with both zoom-in and image search capabilities. When search API keys are available (SERPER_API_KEY and SERPAPI_IMAGE_SEARCH_KEY), this agent can:
- Zoom into specific regions for detailed analysis
- Search for similar images or related information
- Combine both tools for comprehensive image understanding

### Example 3: Custom Image Analysis
Allows users to analyze their own images with custom questions, using the zoom-in capability for detailed examination.

## Dependencies

This script automatically installs the following dependencies:
- qwen-agent
- requests
- pillow

## Usage Requirements

- Local LLM server running at `http://localhost:8080`
- Model must support vision capabilities (image input/output)
- For search functionality: SERPER_API_KEY and SERPAPI_IMAGE_SEARCH_KEY environment variables

## Output

The script processes images and provides:
- Detailed visual analysis with iterative reasoning
- Zoomed-in examination of specific regions
- Structured responses following the thinking process
- Tool usage for enhanced image understanding

## Tool Capabilities

The system includes specialized tools:
- **image_zoom_in_tool**: Allows zooming into specific regions of an image using bbox_2d coordinates in relative format [0, 1000]
- **image_search**: Searches for similar images or related information (requires API keys)

## API Integration

Uses Qwen-Agent framework to communicate with the local server, handling both image and text inputs with function calling capabilities. The system includes proper configuration for generation parameters like temperature, top_p, and repetition_penalty.

## Error Handling

- Includes server connection testing functionality
- Provides informative error messages for missing API keys
- Handles tool unavailability gracefully (e.g., when search keys are missing)
- Validates image accessibility before processing
- Reports specific issues with local server capabilities

## Analysis Framework

The script implements a structured thinking process for image analysis:
1. Detailed visual description of the image
2. Identification of what can be seen just by looking
3. Determination of what requires additional research
4. Tool usage for deeper analysis
5. Synthesis of information into a comprehensive answer

## Customization

Users can customize the script to:
- Analyze their own images with custom questions
- Adjust generation parameters for different analysis styles
- Use different image URLs or local file paths
- Modify the system prompt for specific analysis requirements