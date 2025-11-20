# VL-SDK: Qwen3-VL Examples and Tutorials

This repository contains a collection of examples demonstrating the capabilities of Qwen3-VL, a powerful multimodal language model. Each example showcases different aspects of vision-language understanding and processing, all configured to work with a local LLM server instead of the DashScope API.

## Table of Contents

- [2D Grounding](#2d-grounding)
- [3D Grounding](#3d-grounding)
- [Computer Use](#computer-use)
- [Document Parsing](#document-parsing)
- [Long Document Understanding](#long-document-understanding)
- [Multimodal Coding (MMCode)](#multimodal-coding-mmcode)
- [Mobile Agent](#mobile-agent)
- [OCR](#ocr)
- [Omni Recognition](#omni-recognition)

## Setup Requirements

All examples use a local LLM server configuration. Before running any of the examples, ensure:

1. A local LLM server is running at `http://localhost:8080`
2. The appropriate model is loaded:
   - `Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm` for most examples
   - `Qwen3-Omni-10k` for omni recognition
3. Required assets are downloaded to the appropriate directories under `./assets/`

## Local Configuration

Each example is pre-configured with:
- **API Key**: `dummy-key` (local server may not validate)
- **Base URL**: `http://localhost:8080/v1`
- **Model ID**: Varies by example (see specific example documentation)

## 2D Grounding

### Description
Demonstrates Qwen3-VL's advanced spatial localization abilities, including accurate object detection and specific target grounding within images. Features relative coordinate system (0-1000) and multi-target grounding improvements.

### Source Code
- [2d_grounding.py](2d_grounding.py) - Main implementation with 5 examples

### Documentation
- [2d_grounding.md](2d_grounding.md) - Detailed documentation and configuration

### Output
- Log file: [output/2d_grounding_local.log](output/2d_grounding_local.log) - Complete execution log with API responses and bounding box coordinates
- Output files: None generated (results shown in terminal)

### Examples Included
1. **Multi-Target Object Detection (Dining Table)**: Detects plates, dishes, scallops, wine bottles, bowls, spoons, etc. with bounding boxes in JSON format.
2. **Detecting Objects in Crowded Scenes**: Locates heads, hands, men, women, and glasses in images with many people.
3. **Detecting Objects in Drone View Image**: Identifies cars, buses, bicycles, and pedestrians in aerial imagery.
4. **Detecting Vehicles with Additional Information**: Locates vehicles and returns bounding box coordinates, vehicle type, and color in JSON format.
5. **Pointing out People in Football Field**: Identifies people and returns point coordinates with their roles (player, referee, or unknown) and shirt colors.

## 3D Grounding

### Description
Shows Qwen3-VL's advanced spatial understanding capabilities, including 3D object localization and specific target grounding within images. Works with camera intrinsic parameters for accurate 3D perception.

### Source Code
- [3d_grounding.py](3d_grounding.py) - Main implementation with 4 examples

### Documentation
- [3d_grounding.md](3d_grounding.md) - Detailed documentation and configuration

### Output
- Log file: [output/3d_grounding_local.log](output/3d_grounding_local.log) - Complete execution log with 3D bounding box coordinates and visualization results
- Output files: None generated (results shown in terminal with matplotlib)

### Examples Included
1. **Detect All Cars in Autonomous Driving Scene**: Finds all cars in the image and provides 3D bounding boxes with center coordinates, dimensions, and rotation angles.
2. **Detect Specific Object Using Descriptions**: Locates a specific object (e.g., black chair) in an office image and provides the 3D bounding box.
3. **Detect Multiple Objects Simultaneously**: Identifies multiple furniture objects (tables, chairs, sofas) in Chinese language and outputs their 3D bounding boxes.
4. **Using Custom Camera Parameters**: Detects a bottle in a manipulation scene and predicts its 3D bounding box using generated camera parameters.

## Computer Use

### Description
Demonstrates how to use Qwen3-VL for computer use tasks. Takes a screenshot of a user's desktop and a query, then uses the model to interpret the user's query on the screenshot.

### Source Code
- [computer_use.py](computer_use.py) - Main implementation with 3 examples

### Documentation
- [computer_use.md](computer_use.md) - Detailed documentation and configuration

### Output
- Log file: [output/computer_use_local.log](output/computer_use_local.log) - Complete execution log with coordinate predictions
- Output files: None generated (results shown in terminal)

### Examples Included
1. **Open the Third Issue**: Given a GitHub-style interface screenshot, identifies the specific coordinates of the third issue element.
2. **Reload Cache**: Given a screenshot of a page with a "Reload cache" button, identifies where to click to perform the action.
3. **Open the First Issue**: Given the same GitHub-style interface, identifies the specific coordinates of the first issue element.

## Document Parsing

### Description
Showcases powerful document parsing capabilities that can process any image and output its contents in various formats such as HTML, JSON, Markdown, and LaTeX. Introduces two unique Qwenvl formats with positional information.

### Source Code
- [document_parsing.py](document_parsing.py) - Main implementation with 2 examples

### Documentation
- [document_parsing.md](document_parsing.md) - Detailed documentation and configuration

### Output
- Log file: [output/document_parsing_local.log](output/document_parsing_local.log) - Complete execution log
- Output files: 
  - [image2code_output.html](image2code_output.html) - Generated HTML output from image
  - Temporary files cleaned up after processing

### Examples Included
1. **Document Parsing in QwenVL HTML Format**: Converts an image document to HTML format with bounding box coordinates and positional data-bbox attributes.
2. **Document Parsing in QwenVL Markdown Format**: Converts an image document to specialized QwenVL Markdown format with coordinate information for tables and images.

## Long Document Understanding

### Description
Demonstrates Qwen3-VL's capabilities for understanding long documents with hundreds of pages. Shows how the model can be applied to full-PDF document analysis scenarios.

### Source Code
- [long_document_understanding.py](long_document_understanding.py) - Main implementation with 2 examples

### Documentation
- [long_document_understanding.md](long_document_understanding.md) - Detailed documentation and configuration

### Output
- Log file: [output/long_document_understanding_local.log](output/long_document_understanding_local.log) - Complete execution log with PDF processing results
- Output files: None generated (results shown in terminal)

### Examples Included
1. **Long Document Summarization**: Analyzes a long PDF document and provides a summary of key contributions based on the abstract and introduction.
2. **Counting Tables in a Document**: Analyzes a PDF document and counts the number of tables present.

## Multimodal Coding (MMCode)

### Description
Demonstrates three key capabilities of Qwen3-VL for multimodal coding tasks: Image to HTML, Chart to Code, and Multimodal Coding Challenges.

### Source Code
- [mmcode.py](mmcode.py) - Main implementation with 3 examples

### Documentation
- [mmcode.md](mmcode.md) - Detailed documentation and configuration

### Output
- Log file: [output/mmcode_local.log](output/mmcode_local.log) - Complete execution log
- Output files:
  - [image2code_output.html](image2code_output.html) - Generated HTML from image-to-code example
  - [output/mmcode_solution.txt](output/mmcode_solution.txt) - Generated solution for MMCode problems

### Examples Included
1. **Image-to-HTML Conversion**: Converts a screenshot or sketch image into clean, functional, and modern HTML code.
2. **Chart-to-Code**: Analyzes chart images and generates corresponding Python matplotlib code that can reproduce the chart.
3. **Multimodal Coding Challenges**: Solves programming problems from the MMCode benchmark that require both visual and text understanding.

## Mobile Agent

### Description
Demonstrates Qwen3-VL's agent function call capabilities to interact with a mobile device. Shows the model's ability to generate and execute actions based on user queries and visual context.

### Source Code
- [mobile_agent.py](mobile_agent.py) - Main implementation with 2 examples

### Documentation
- [mobile_agent.md](mobile_agent.md) - Detailed documentation and configuration

### Output
- Log file: [output/mobile_agent_local.log](output/mobile_agent_local.log) - Complete execution log with action plans
- Output files: None generated (results shown in terminal)

### Examples Included
1. **English App & Query with Local API**: Given a screenshot of an English mobile app interface, tasked with searching for Musk in X and navigating to his homepage.
2. **Chinese App & Query with Local API**: Given a screenshot of a Chinese mobile app interface, tasked with searching for Musk in X and navigating to his homepage.

## OCR

### Description
Demonstrates Qwen3-VL's OCR capabilities: full-page OCR for English and multilingual text, text spotting with bounding boxes, and visual information extraction.

### Source Code
- [ocr.py](ocr.py) - Main implementation with 5 examples

### Documentation
- [ocr.md](ocr.md) - Detailed documentation and configuration

### Output
- Log file: [output/ocr_local.log](output/ocr_local.log) - Complete execution log with OCR results
- Output files: None generated (results shown in terminal)

### Examples Included
1. **Full-page OCR for English Text**: Reads all text in the image with a simple prompt.
2. **Full-page OCR for Multilingual Text**: Extracts text content from images containing multilingual text.
3. **Text Spotting (Line-level)**: Detects text with line-level granularity and outputs bounding box coordinates in JSON format.
4. **Text Spotting (Word-level)**: Detects text with word-level granularity and outputs bounding box coordinates in JSON format.
5. **Visual Information Extraction**: Extracts specific key-value information from structured documents like receipts and invoices.

## Omni Recognition

### Description
Demonstrates Qwen3-VL's omni recognition capabilities: object recognition (celebrities, anime characters, landmarks, animals, plants), object spotting with bounding boxes, and multi-object identification.

### Source Code
- [omni_recognition.py](omni_recognition.py) - Main implementation with 3 examples

### Documentation
- [omni_recognition.md](omni_recognition.md) - Detailed documentation and configuration

### Output
- Log file: [output/omni_recognition_local.log](output/omni_recognition_local.log) - Complete execution log with recognition results
- Output files: None generated (results shown in terminal)

### Examples Included
1. **Object Recognition**: Recognizes celebrities, anime characters, landmarks, animals, plants, etc. with queries like "Who is this?" or "What kind of bird is this?".
2. **Object Spotting with Bounding Boxes**: Identifies multiple objects and provides their coordinates in JSON format with names in both Chinese and English.
3. **Multi-object Recognition**: Performs comprehensive scene analysis to identify all recognizable objects in a complex image with descriptions.

## Other Examples

Additional examples not detailed above:
- [spatial_understanding.md](spatial_understanding.md) and [spatial_understanding.py](spatial_understanding.py)
- [think_with_images.md](think_with_images.md) and [think_with_images.py](think_with_images.py)
- [video_understanding.md](video_understanding.md) and [video_understanding.py](video_understanding.py)

## Additional Resources

- [mmcode_test.jsonl.gz](mmcode_test.jsonl.gz) - MMCode benchmark dataset used in multimodal coding challenges
- Various assets in `./assets/` subdirectories used by the examples
- Cache directory for PDF processing in long document understanding example

## Contributing

Each example is designed to be self-contained and run independently. The examples have been modified to use local LLM servers instead of the DashScope API, making them more accessible for local development and experimentation.