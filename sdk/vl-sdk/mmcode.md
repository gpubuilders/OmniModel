# Multimodal Coding with Qwen3-VL (Local Version)

This example demonstrates three key capabilities of Qwen3-VL for multimodal coding tasks:

1. **Image to HTML**: Convert screenshots or sketches into functional HTML code
2. **Chart to Code**: Analyze chart images and generate corresponding plotting code
3. **Multimodal Coding Challenges**: Solve programming problems that require visual understanding

## Key Features

- **Image-to-Code Conversion**: Converts visual inputs to functional code in various formats
- **Multimodal Problem Solving**: Solves programming challenges that combine text and image inputs
- **MMCode Challenge Integration**: Supports the MMCode benchmark for multimodal coding evaluation
- **Local API Integration**: Uses a local LLM server with OpenAI-compatible API endpoint

## Local Configuration

- **API Key**: `dummy-key` (local server may not validate)
- **Base URL**: `http://localhost:8080/v1`
- **Model ID**: `Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm`

## Examples Included

### Example 1: Image-to-HTML Conversion
Converts a screenshot or sketch image into clean, functional, and modern HTML code. The model analyzes the visual layout and generates appropriate HTML structure with styling.

### Example 2: Chart-to-Code
Analyzes chart images and generates corresponding Python matplotlib code that can reproduce the chart. The model understands visual data representation and translates it to code.

### Example 3: Multimodal Coding Challenges
Solves programming problems from the MMCode benchmark that require both visual and text understanding. The model processes interleaved image and text inputs to generate Python solutions.

## Dependencies

This script automatically installs the following dependencies:
- Pillow
- requests
- openai
- matplotlib
- numpy
- playwright

## Usage Requirements

- Local LLM server running at `http://localhost:8080`
- Required assets in `./assets/multimodal_coding/` directory:
  - `screenshot_demo.png` (for image-to-HTML example)
  - `chart2code_input.png` (for chart-to-code example)
- Optional: MMCode test set (mmcode_test.jsonl.gz) for multimodal coding challenges

## Output

The script processes multimodal inputs and provides:
- Generated HTML code from visual inputs
- Python matplotlib code from chart images
- Solutions to multimodal programming problems
- Safe execution environment with configurable execution settings

## Code Extraction

The script includes utilities to extract code blocks from model responses using regex patterns for different code formats (Python, HTML). It identifies the last code block in the response for processing.

## Image Processing

The script handles image processing in multiple formats:
- Local file paths for image-to-code conversion
- Base64 encoding for API transmission
- PIL image conversion for internal processing
- Size control with configurable maximum width

## Problem Solving Framework

For multimodal coding challenges, the script:
- Downloads and loads the MMCode test set from HuggingFace
- Processes interleaved image and text inputs following MMCode format
- Generates solutions with appropriate code formatting
- Supports both Standard IO and Call-Based formats for solution generation

## API Integration

Uses OpenAI-compatible client to communicate with the local server, handling both text and image inputs. Includes configurable token limits and temperature settings for different types of requests.

## Error Handling

- Includes API connection testing functionality
- Provides informative error messages if assets are missing
- Handles code generation and execution safely with configurable settings
- Handles both Jupyter and terminal environments appropriately
- Validates code block extraction from model responses

## Safety Features

- Configurable execution of generated code (disabled by default for safety)
- Maximum token limits to prevent excessive output
- Validation of code block extraction before execution
- Safe handling of user-configurable parameters