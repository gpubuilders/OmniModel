# Long Document Understanding with Qwen3-VL (Local Version)

This example demonstrates Qwen3-VL's capabilities for understanding long documents with hundreds of pages. It showcases how the advanced model can be applied to full-PDF document analysis scenarios using a local LLM server instead of the DashScope API.

## Key Features

- **Long Document Processing**: Capable of processing PDF documents with hundreds of pages
- **Visual Grid Display**: Creates thumbnail grids to visualize document content efficiently
- **Image Caching**: Implements caching to avoid redundant PDF-to-image conversions
- **Local API Integration**: Uses a local LLM server with OpenAI-compatible API endpoint

## Local Configuration

- **API Key**: `dummy-key` (local server may not validate)
- **Base URL**: `http://localhost:8080/v1`
- **Model ID**: `Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm`

## Examples Included

### Example 1: Long Document Summarization
Analyzes a long PDF document (Qwen2.5-VL paper) and provides a summary of key contributions based on the abstract and introduction. The script converts the PDF to images and sends them to the model for analysis.

### Example 2: Counting Tables in a Document
Analyzes another PDF document (Fox's "Got Merge Code" paper) and counts the number of tables present. This demonstrates the model's ability to identify and count specific elements across a long document.

## Dependencies

This script automatically installs the following dependencies:
- Pillow
- requests
- openai
- qwen-vl-utils
- pdf2image
- torch
- transformers

Note: Requires poppler-utils to be installed on the system for PDF processing.

## Usage Requirements

- Local LLM server running at `http://localhost:8080`
- Required assets in `./assets/long_document_understanding/` directory:
  - `Qwen2.5-VL.pdf`
  - `fox_got_merge_code.pdf`
- System-level poppler-utils package for PDF processing

## Output

The script processes long PDF documents and provides:
- Thumbnail grid visualizations of document pages
- Text analysis responses for document content
- Efficient caching of PDF-to-image conversions
- Coordinate and sizing information for document elements

## PDF Processing

The script includes:
- PDF-to-image conversion with configurable DPI (default 144)
- Image caching based on PDF hash to avoid redundant processing
- Image size control to maintain maximum side length of 1500 pixels
- Efficient thumbnail grid creation for document overview

## API Integration

Uses OpenAI-compatible client to communicate with the local server, handling batch processing of multiple images from PDF documents. The script encodes each image to base64 format for transmission to the API server.

## Error Handling

- Includes API connection testing functionality
- Provides informative error messages if PDF files are missing
- Handles PDF conversion failures gracefully
- Handles both Jupyter and terminal environments appropriately
- Provides feedback on caching and processing status

## Caching System

Implements an efficient caching system that:
- Creates unique hashes for PDF files based on their path
- Stores converted images as numpy arrays in the cache directory
- Reuses cached images on subsequent runs to avoid redundant conversions
- Reports cache usage statistics for transparency