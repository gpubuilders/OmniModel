# Video Understanding with Qwen3-VL (Local Version)

This example demonstrates Qwen3-VL's capabilities for video understanding tasks using a local LLM server. It showcases various approaches for processing video content including URL-based analysis, frame-by-frame processing, and spatial-temporal grounding.

## Key Features

- **Video URL Analysis**: Process video content directly from URLs
- **Frame List Processing**: Analyze pre-extracted video frames
- **Spatial-Temporal Grounding**: Localize objects and events across both space and time
- **Event Localization**: Identify and timestamp specific activities in videos
- **Interleaved Timestamp-Image Processing**: Combine temporal and visual information
- **Local API Integration**: Uses a local LLM server with OpenAI-compatible API endpoint

## Local Configuration

- **API Key**: `dummy-key` (local server may not validate)
- **Base URL**: `http://localhost:8080/v1`
- **Model ID**: `Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm`

## Examples Included

### Example 1: Video URL Analysis (Chinese)
Analyzes a video from a URL and summarizes product features in a table format. Demonstrates basic video understanding with frame extraction and visualization.

### Example 2: Video Event Localization with Timestamps
Localizes activity events in a video and outputs start and end timestamps for each event with descriptive sentences. Results are provided in JSON format with 'mm:ss.ff' time format.

### Example 3: Frame List Analysis (Cooking Video)
Processes pre-extracted frames from a cooking video at 0.25 FPS (1 frame per 4 seconds) to provide a brief description of the video content.

### Example 4: Frame List Analysis (Math Video)
Analyzes frames from a math instruction video at 0.5 FPS to describe the educational content being presented.

### Example 5: Spatial-Temporal Grounding
Uses interleaved timestamp-image pairs to detect and localize specific objects mentioned in a query (e.g., "moving bicycle towards an adult in black") across multiple video frames.

## Dependencies

This script automatically installs the following dependencies:
- openai (through the script)
- Additional dependencies handled via imports:
  - decord (for video processing)
  - numpy
  - pillow
  - requests
  - markdown
  - beautifulsoup4

## Usage Requirements

- Local LLM server running at `http://localhost:8080`
- Internet access to download remote video files for processing
- Model must support video understanding capabilities
- For frame processing: decord library for video frame extraction

## Input Methods

The script supports multiple video input formats:

1. **Video URL**: Direct URL to video files (MP4, AVI, etc.) processed internally by the model
2. **Frame List**: Pre-extracted list of PIL Image objects or image URLs representing sampled frames
3. **Interleaved Pairs**: Timestamp-image combinations for precise temporal-spatial analysis

## Output

The script processes videos and provides:
- Descriptive summaries of video content
- Timestamped event localization in JSON format
- Bounding box detection for specific objects across frames
- Frame grid visualizations for video overview
- Structured analysis results in JSON format

## Video Processing

The script includes comprehensive video processing capabilities:
- Frame extraction at configurable frame rates (e.g., 0.25 FPS)
- Caching mechanisms to avoid redundant processing
- Image grid creation for frame visualization
- Timestamp preservation for temporal analysis

## Frame Extraction

The system includes:
- Configurable frame sampling rates
- Video download and caching functionality
- Frame timestamp extraction and preservation
- Efficient batch processing of frame sequences

## API Integration

Uses OpenAI-compatible client to communicate with the local server, handling both video URL and frame list inputs. The system includes proper message formatting for video-specific API requests.

## Error Handling

- Includes API connection testing functionality
- Provides informative error messages for network issues
- Handles video download and processing failures gracefully
- Validates model capabilities before processing
- Reports specific video processing errors

## Visualization

The script includes specialized functions to:
- Create image grids from video frames for overview visualization
- Draw bounding boxes on frames for spatial-temporal detection results
- Display frame sequences in temporal order
- Show detection results with timestamp alignment