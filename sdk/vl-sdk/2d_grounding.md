# 2D Grounding with Qwen3-VL (Local Version)

This example demonstrates Qwen3-VL's advanced spatial localization abilities, including accurate object detection and specific target grounding within images using a local LLM server instead of the DashScope API.

## Key Features

- **Coordinate System**: Qwen3-VL's default coordinate system has been changed from absolute coordinates used in Qwen2.5-VL to relative coordinates ranging from 0 to 1000
- **Multi-Target Grounding**: Qwen3-VL has improved its multi-target grounding ability
- **Local API Integration**: Uses a local LLM server with OpenAI-compatible API endpoint

## Local Configuration

- **API Key**: `dummy-key` (local server may not validate)
- **Base URL**: `http://localhost:8080/v1`
- **Model ID**: `Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm`

## Examples Included

### Example 1: Multi-Target Object Detection (Dining Table)
Detects multiple objects like plates, dishes, scallops, wine bottles, bowls, spoons, etc., with bounding boxes in JSON format.

### Example 2: Detecting Objects in Crowded Scenes
Locates heads, hands, men, women, and glasses in images with many people, returning bounding box coordinates in JSON format.

### Example 3: Detecting Objects in Drone View Image
Identifies cars, buses, bicycles, and pedestrians in aerial imagery, returning bounding box coordinates in JSON format.

### Example 4: Detecting Vehicles with Additional Information
Locates vehicles and returns not only bounding box coordinates but also vehicle type and color in a structured JSON format.

### Example 5: Pointing out People in Football Field
Identifies people in a football field image and returns point coordinates with their roles (player, referee, or unknown) and shirt colors in JSON format.

## Dependencies

This script automatically installs the following dependencies:
- Pillow
- requests
- openai

## Usage Requirements

- Local LLM server running at `http://localhost:8080`
- Required assets in `./assets/spatial_understanding/` directory:
  - `dining_table.png`
  - `lots_of_people.jpeg`
  - `lots_of_cars.png`
  - `drone_cars2.png`
  - `football_field.jpg`

## Output

The script processes images and visualizes results by:
- Drawing bounding boxes around detected objects
- Using different colors for different object types
- Displaying labels for each detected object
- Providing coordinates in JSON format

## Visualization

The script includes utility functions to plot bounding boxes and points on images, handling both Jupyter and terminal environments. It uses PIL for image processing and can handle different coordinate systems for accurate object localization.

## API Integration

Uses OpenAI-compatible client to communicate with the local server, handling both image and text inputs. The script automatically converts local images to base64 format for transmission to the API server.

## Error Handling

- Includes API connection testing functionality
- Provides informative error messages if assets are missing
- Handles both Jupyter and terminal environments appropriately
- Provides fallback to default fonts if specialized fonts are not available