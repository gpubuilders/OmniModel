# 3D Grounding with Qwen3-VL (Local Version)

This example demonstrates Qwen3-VL's advanced spatial understanding capabilities, including 3D object localization and specific target grounding within images using a local LLM server instead of the DashScope API.

## Key Features

- **3D Object Localization**: Qwen3-VL supports localizing specific 3D objects in images based on natural language descriptions
- **Camera Parameter Integration**: Works with real camera intrinsic parameters for accurate 3D perception
- **Multi-Object Detection**: Capable of detecting and localizing multiple 3D objects simultaneously
- **Local API Integration**: Uses a local LLM server with OpenAI-compatible API endpoint

## Local Configuration

- **API Key**: `dummy-key` (local server may not validate)
- **Base URL**: `http://localhost:8080/v1`
- **Model ID**: `Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm`

## 3D Bounding Box Format

The 3D bounding boxes are represented as: `[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]`

- **x_center, y_center, z_center**: Object center in camera coordinates (meters)
- **x_size, y_size, z_size**: Object dimensions (meters)
- **roll, pitch, yaw**: Rotation angles (radians, multiplied by 180)

## Examples Included

### Example 1: Detect All Cars in Autonomous Driving Scene
Finds all cars in the image and provides 3D bounding boxes with center coordinates, dimensions, and rotation angles for each car in JSON format.

### Example 2: Detect Specific Object Using Descriptions
Locates a specific object (e.g., black chair) in an office image and provides the 3D bounding box in JSON format.

### Example 3: Detect Multiple Objects Simultaneously
Identifies multiple furniture objects (tables, chairs, sofas) in Chinese language and outputs their 3D bounding boxes in JSON format.

### Example 4: Using Custom Camera Parameters
Detects a bottle in a manipulation scene and predicts its 3D bounding box using generated camera parameters when original parameters are unavailable.

## Dependencies

This script automatically installs the following dependencies:
- Pillow
- requests
- openai
- opencv-python
- numpy
- matplotlib

## Usage Requirements

- Local LLM server running at `http://localhost:8080`
- Required assets in `./assets/spatial_understanding/` directory:
  - `autonomous_driving.jpg`
  - `office.jpg`
  - `lounge.jpg`
  - `manipulation.jpg`
  - `cam_infos.json` (for camera parameters; default parameters generated if not available)

## Output

The script processes images and visualizes 3D results by:
- Drawing 3D bounding boxes projected onto the 2D image
- Using multiple random colors for different objects
- Creating matplotlib figures showing the annotated images
- Providing 3D bounding box coordinates in JSON format

## 3D Visualization

The script includes comprehensive 3D visualization utilities that:
- Convert 3D bounding boxes to 2D image coordinates using camera parameters
- Render 3D boxes with 12 edges connecting 8 corners
- Handle rotation transformations using pitch, yaw, and roll angles
- Support both loaded camera parameters and generated default parameters

## Camera Parameters

The script supports both:
- Loading specific camera intrinsic parameters from `cam_infos.json` (fx, fy, cx, cy)
- Generating default parameters based on image dimensions and field of view (60° default) when specific parameters are unavailable

## API Integration

Uses OpenAI-compatible client to communicate with the local server, handling image encoding to base64 format for transmission to the API server.

## Error Handling

- Includes API connection testing functionality
- Provides informative error messages if assets are missing
- Handles both Jupyter and terminal environments appropriately
- Generates fallback camera parameters when original parameters are unavailable