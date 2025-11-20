# Spatial Understanding with Qwen3-VL (Local Version)

This example demonstrates Qwen3-VL's ability to do more than just see objects. It understands their spatial layout, perceives what actions are possible ('affordances'), and uses this knowledge to reason like an embodied agent using a local LLM service.

## Key Features

- **Spatial Relationship Understanding**: Analyzes relationships between objects in 3D space
- **Object Affordance Perception**: Understands what actions are possible with objects in the environment
- **Action Planning**: Integrates spatial reasoning with action planning for robotic applications
- **Video Spatial Reasoning**: Extends spatial understanding to video sequences
- **Local API Integration**: Uses a local LLM server with OpenAI-compatible API endpoint

## Local Configuration

- **API Key**: `dummy-key` (local server may not validate)
- **Base URL**: `http://localhost:8080/v1`
- **Model ID**: `Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm`

## Examples Included

### Example 1: Spatial Relationships Between Objects
Analyzes the relative positioning of objects in an image and answers questions about spatial relationships, such as identifying which object is farthest from the viewer's position.

### Example 2: Object Affordances
Perceives what actions are possible with objects in the scene. This includes identifying free spaces, determining if objects can fit in certain locations, and understanding physical constraints.

### Example 3: Spatial Reasoning and Action Planning
Integrates spatial understanding with action planning for robotics applications. Determines the appropriate actions to achieve spatial goals, such as moving objects to specific locations.

### Example 4: Video Spatial Reasoning
Extends spatial reasoning to video sequences, analyzing how spatial relationships change over time and planning actions based on dynamic scenes.

## Dependencies

This script automatically installs the following dependencies:
- Pillow
- numpy
- openai
- decord

## Usage Requirements

- Local LLM server running at `http://localhost:8080`
- Required assets in `./assets/spatial_understanding/` directory:
  - `spatio_case1.jpg` (for spatial relationships)
  - `spatio_case2_aff.png` (for affordances with point coordinates)
  - `spatio_case2_aff2.png` (for second affordance example)
  - `spatio_case2_plan.png` (for action planning)
  - `spatio_case2_plan2.png` (for second action planning example)

## Output

The script processes spatial images and provides:
- Analysis of spatial relationships between objects
- Point coordinates for specific locations in images
- Action recommendations based on spatial constraints
- Visual annotations showing recognized spatial features

## Spatial Reasoning Capabilities

The spatial understanding functionality includes:
- Relative positioning analysis (farthest, closest, etc.)
- Point-based location identification with coordinates
- Affordance detection (what actions are possible)
- Action planning based on spatial constraints
- Multi-frame spatial reasoning for video inputs

## Coordinate System

Point-based spatial queries use a 0-1000 relative coordinate system. The script converts these relative coordinates to absolute pixel positions based on the image dimensions for proper visualization.

## Visualization

The script includes specialized functions to:
- Plot points at specified coordinates on spatial understanding images
- Scale coordinates appropriately for different image dimensions
- Visualize spatial relationships with colored markers and labels
- Show the current spatial configuration before spatial reasoning

## API Integration

Uses OpenAI-compatible client to communicate with the local server, handling both single image and video inputs. The system includes proper base64 encoding for image transmission to the API server.

## Error Handling

- Includes API connection testing functionality
- Provides informative error messages if assets are missing
- Handles coordinate parsing failures for spatial point outputs
- Handles both Jupyter and terminal environments appropriately
- Gracefully degrades when video assets are not available

## Video Processing

For video-based spatial reasoning, the script includes:
- Frame extraction and caching using decord
- Video download and local storage management
- Frame grid visualization for temporal analysis
- Proper handling of video-specific API requests