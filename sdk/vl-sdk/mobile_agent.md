# Mobile Agent with Qwen3-VL (Local Version)

This example demonstrates how to use Qwen3-VL's agent function call capabilities to interact with a mobile device. It showcases the model's ability to generate and execute actions based on user queries and visual context using a local LLM server instead of the DashScope API.

## Key Features

- **Mobile Device Interaction**: Generates actions for touchscreen mobile device interaction
- **Function Calling**: Uses structured JSON function calls to perform mobile actions
- **Visual Action Planning**: Analyzes screenshots to determine appropriate actions
- **Local API Integration**: Uses a local LLM server with OpenAI-compatible API endpoint

## Local Configuration

- **API Key**: `dummy-key` (local server may not validate)
- **Base URL**: `http://localhost:8080/v1`
- **Model ID**: `Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm`

## Supported Actions

The mobile agent can perform the following actions:
- **click**: Click at specified coordinates (x, y)
- **long_press**: Press and hold at specified coordinates for a duration
- **swipe**: Swipe from one coordinate to another
- **type**: Input text into an activated input box
- **answer**: Output a text answer
- **system_button**: Press system buttons (Back, Home, Menu, Enter)
- **wait**: Wait for a specified number of seconds
- **terminate**: End the task with success or failure status

## Examples Included

### Example 1: English App & Query with Local API
Given a screenshot of an English mobile app interface, the model is tasked with searching for "Musk" in X (Twitter) and navigating to his homepage to open the first post. The system has already opened the X app.

### Example 2: Chinese App & Query with Local API
Given a screenshot of a Chinese mobile app interface (Bilibili), the model is tasked with searching for "Musk" in X and navigating to his homepage to open the first post. The system has already performed several search-related actions.

## Dependencies

This script automatically installs the following dependencies:
- Pillow
- requests
- openai
- icecream
- matplotlib

## Usage Requirements

- Local LLM server running at `http://localhost:8080`
- Required assets in `./assets/agent_function_call/` directory:
  - `mobile_en_example.png` (for English app example)
  - `mobile_zh_example.png` (for Chinese app example)

## Output

The script processes mobile screenshots and provides:
- JSON-formatted action plans with coordinates and parameters
- Visual annotations showing where to click (green circles at coordinates)
- Action history tracking with previous steps
- System prompt integration with mobile device specifications

## Coordinate System

The mobile device uses a 999x999 coordinate system. The script includes utilities to rescale coordinates from the model's coordinate system to the actual image dimensions for accurate visualization.

## Mobile Interaction Framework

The system includes:
- Screenshot analysis for current mobile state
- History tracking of previous actions
- System prompt with mobile-specific tools and instructions
- Coordinate-based action execution on the mobile interface

## Visualization

The script includes specialized functions to:
- Draw points at specified coordinates on mobile screenshots
- Scale coordinates appropriately for different image dimensions
- Visualize planned actions with green circles for clarity
- Show the current mobile state before action execution

## API Integration

Uses OpenAI-compatible client to communicate with the local server, handling both system and user messages along with image inputs. The system includes structured message formatting for function calling.

## Error Handling

- Includes API connection testing functionality
- Provides informative error messages if assets are missing
- Handles JSON parsing failures for function calls
- Handles both Jupyter and terminal environments appropriately
- Provides fallback visualization when coordinate parsing fails