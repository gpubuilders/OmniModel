#!/usr/bin/env python3
# coding: utf-8

"""
OCR with Qwen3-VL (Local Version)

This script demonstrates Qwen3-VL's OCR capabilities using a local LLM server:
- Full-page OCR for English and multilingual text
- Text spotting with bounding boxes
- Visual information extraction

All dependencies will be auto-installed if missing.
"""

import subprocess
import sys
import os

# Auto-install function
def install_package(package):
    """Install a package if it's not already installed"""
    try:
        __import__(package.split('[')[0])
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", package])
        print(f"✓ {package} installed successfully")

# Install required packages
print("="*70)
print("Checking and installing dependencies...")
print("="*70)

required_packages = [
    "requests",
    "pillow",
]

for package in required_packages:
    install_package(package)

print("\n✓ All dependencies installed!\n")

# Now import everything
import requests
import json
import base64
import ast
from io import BytesIO
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

# Local LLM Configuration
LOCAL_MODEL_ID = 'Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm'
LOCAL_API_BASE = 'http://localhost:8080/v1'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def encode_image_to_base64(image_path):
    """Encode an image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_json_output(json_output):
    """Parse JSON output, removing markdown fencing if present"""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output.strip()


def inference_local(image_path, prompt, system_prompt="You are a helpful assistant.", max_tokens=4096):
    """
    Run inference on local LLM server with vision capabilities
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt for the model
        system_prompt: System message (optional)
        max_tokens: Maximum tokens to generate
        
    Returns:
        str: Model response text
    """
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    # Make API call
    try:
        response = requests.post(
            f"{LOCAL_API_BASE}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": LOCAL_MODEL_ID,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.8,
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"API returned status {response.status_code}: {response.text}")
            
    except Exception as e:
        raise Exception(f"Inference failed: {e}")


def display_text_boxes_terminal(image_path, bounding_boxes):
    """
    Display bounding boxes in terminal format (text-based representation)
    
    Args:
        image_path: Path to the image
        bounding_boxes: JSON string with bounding box data
    """
    # Load image to get dimensions
    img = Image.open(image_path)
    width, height = img.size
    
    print(f"\nImage dimensions: {width}x{height}")
    print("-" * 70)
    
    # Parse JSON
    bounding_boxes = parse_json_output(bounding_boxes)
    
    try:
        boxes = ast.literal_eval(bounding_boxes)
        
        print(f"\nFound {len(boxes)} text regions:\n")
        
        for i, box in enumerate(boxes, 1):
            # Convert normalized coordinates (0-999) to absolute
            abs_x1 = int(box["bbox_2d"][0] / 999 * width)
            abs_y1 = int(box["bbox_2d"][1] / 999 * height)
            abs_x2 = int(box["bbox_2d"][2] / 999 * width)
            abs_y2 = int(box["bbox_2d"][3] / 999 * height)
            
            # Ensure correct order
            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1
            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1
            
            text = box.get("text_content", "")
            
            print(f"[{i}] Text: {text}")
            print(f"    Position: ({abs_x1}, {abs_y1}) → ({abs_x2}, {abs_y2})")
            print(f"    Size: {abs_x2-abs_x1}x{abs_y2-abs_y1} pixels")
            print()
            
    except Exception as e:
        print(f"Error parsing bounding boxes: {e}")
        print(f"Raw output:\n{bounding_boxes}")


# ============================================================================
# TEST CONNECTION
# ============================================================================

def test_connection():
    """Test if the local LLM server is accessible"""
    print("\n" + "="*70)
    print("Testing Local LLM Connection")
    print("="*70)
    
    try:
        response = requests.get(f"{LOCAL_API_BASE.replace('/v1', '')}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"\n✓ Server is accessible at {LOCAL_API_BASE}")
            
            # Check if our model is available
            model_found = False
            if 'data' in models:
                print("\nAvailable models:")
                for model in models['data']:
                    model_id = model.get('id', 'unknown')
                    if model_id == LOCAL_MODEL_ID:
                        print(f"  ✓ {model_id} (target model)")
                        model_found = True
                    else:
                        print(f"    {model_id}")
            
            if model_found:
                print(f"\n✓ Target model '{LOCAL_MODEL_ID}' is available!")
                return True
            else:
                print(f"\n⚠ Warning: Model '{LOCAL_MODEL_ID}' not found in server")
                print("The script will continue but may fail if the model doesn't match.")
                return True  # Continue anyway
        else:
            print(f"\n✗ Server returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Cannot connect to server at {LOCAL_API_BASE}")
        print("\nPlease ensure:")
        print("1. Your local LLM server is running")
        print("2. The server is accessible at http://localhost:8080")
        print("3. The /v1/models endpoint is available")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


# ============================================================================
# OCR EXAMPLES
# ============================================================================

def example_1_full_page_ocr_english(image_path):
    """
    Example 1: Full-page OCR for English text
    
    Args:
        image_path: Path to the image containing text
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Full-page OCR for English Text")
    print("="*70)
    
    prompt = "Read all the text in the image."
    
    print(f"\nImage: {image_path}")
    print(f"Prompt: {prompt}")
    print("\n" + "-"*70)
    print("OCR RESULT:")
    print("-"*70 + "\n")
    
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"✗ Error: Image not found at {image_path}")
            return None
            
        response = inference_local(image_path, prompt)
        print(response)
        print("\n" + "="*70 + "\n")
        return response
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def example_2_full_page_ocr_multilingual(image_path):
    """
    Example 2: Full-page OCR for multilingual text
    
    Args:
        image_path: Path to the image containing multilingual text
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Full-page OCR for Multilingual Text")
    print("="*70)
    
    prompt = "Please output only the text content from the image without any additional descriptions or formatting."
    
    print(f"\nImage: {image_path}")
    print(f"Prompt: {prompt}")
    print("\n" + "-"*70)
    print("OCR RESULT:")
    print("-"*70 + "\n")
    
    try:
        if not os.path.exists(image_path):
            print(f"✗ Error: Image not found at {image_path}")
            return None
            
        response = inference_local(image_path, prompt)
        print(response)
        print("\n" + "="*70 + "\n")
        return response
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def example_3_text_spotting_line_level(image_path):
    """
    Example 3: Text spotting with bounding boxes (line-level)
    
    Args:
        image_path: Path to the image
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Text Spotting (Line-level)")
    print("="*70)
    print("\nThis example detects text and provides bounding box coordinates.")
    print("Coordinates are in relative format [x1, y1, x2, y2] ranging from 0-999.\n")
    
    prompt = "Spotting all the text in the image with line-level, and output in JSON format as [{'bbox_2d': [x1, y1, x2, y2], 'text_content': 'text'}, ...]."
    
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print("\n" + "-"*70)
    print("TEXT SPOTTING RESULT:")
    print("-"*70 + "\n")
    
    try:
        if not os.path.exists(image_path):
            print(f"✗ Error: Image not found at {image_path}")
            return None
            
        response = inference_local(image_path, prompt, max_tokens=8192)
        print("Raw JSON output:")
        print(response)
        print("\n" + "-"*70)
        
        # Display in terminal-friendly format
        display_text_boxes_terminal(image_path, response)
        
        print("="*70 + "\n")
        return response
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def example_4_text_spotting_word_level(image_path):
    """
    Example 4: Text spotting with bounding boxes (word-level)
    
    Args:
        image_path: Path to the image
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Text Spotting (Word-level)")
    print("="*70)
    
    prompt = "Spotting all the text in the image with word-level, and output in JSON format as [{'bbox_2d': [x1, y1, x2, y2], 'text_content': 'text'}, ...]."
    
    print(f"\nImage: {image_path}")
    print(f"Prompt: {prompt}")
    print("\n" + "-"*70)
    print("TEXT SPOTTING RESULT:")
    print("-"*70 + "\n")
    
    try:
        if not os.path.exists(image_path):
            print(f"✗ Error: Image not found at {image_path}")
            return None
            
        response = inference_local(image_path, prompt, max_tokens=8192)
        print("Raw JSON output:")
        print(response)
        print("\n" + "-"*70)
        
        # Display in terminal-friendly format
        display_text_boxes_terminal(image_path, response)
        
        print("="*70 + "\n")
        return response
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def example_5_visual_information_extraction(image_path, keys):
    """
    Example 5: Visual information extraction with given keys
    
    Args:
        image_path: Path to the image (e.g., receipt, invoice)
        keys: Dictionary of keys to extract or list of key names
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Visual Information Extraction")
    print("="*70)
    
    # Format the prompt based on keys type
    if isinstance(keys, dict):
        keys_json = json.dumps(keys, ensure_ascii=False)
        prompt = f"Extract the key-value information in the format: {keys_json}"
    elif isinstance(keys, list):
        prompt = f"提取图中的：{keys}，并且按照json格式输出。"
    else:
        keys_json = json.dumps(keys, ensure_ascii=False)
        prompt = f"Extract the key-value information in the format: {keys_json}"
    
    print(f"\nImage: {image_path}")
    print(f"Keys to extract: {keys}")
    print(f"Prompt: {prompt}")
    print("\n" + "-"*70)
    print("EXTRACTION RESULT:")
    print("-"*70 + "\n")
    
    try:
        if not os.path.exists(image_path):
            print(f"✗ Error: Image not found at {image_path}")
            return None
            
        response = inference_local(image_path, prompt, max_tokens=4096)
        print(response)
        
        # Try to parse and pretty-print JSON
        try:
            parsed = parse_json_output(response)
            json_data = json.loads(parsed)
            print("\n" + "-"*70)
            print("Parsed JSON:")
            print("-"*70)
            print(json.dumps(json_data, indent=2, ensure_ascii=False))
        except:
            pass  # If parsing fails, just show raw output
        
        print("\n" + "="*70 + "\n")
        return response
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def example_custom_ocr(image_path, prompt):
    """
    Custom OCR with your own image and prompt
    
    Args:
        image_path: Path to your image
        prompt: Your custom OCR prompt
    """
    print("\n" + "="*70)
    print("CUSTOM OCR")
    print("="*70)
    
    print(f"\nImage: {image_path}")
    print(f"Prompt: {prompt}")
    print("\n" + "-"*70)
    print("RESULT:")
    print("-"*70 + "\n")
    
    try:
        if not os.path.exists(image_path):
            print(f"✗ Error: Image not found at {image_path}")
            return None
            
        response = inference_local(image_path, prompt, max_tokens=8192)
        print(response)
        print("\n" + "="*70 + "\n")
        return response
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*70)
    print("OCR with Qwen3-VL - Local LLM Version")
    print("="*70)
    print(f"Model: {LOCAL_MODEL_ID}")
    print(f"Server: {LOCAL_API_BASE}")
    print("="*70)
    
    # Test connection first
    if not test_connection():
        print("\n" + "="*70)
        print("CONNECTION FAILED - Cannot proceed")
        print("="*70)
        return
    
    print("\n" + "="*70)
    print("Ready to run OCR examples!")
    print("="*70)
    
    # Check if assets directory exists
    assets_dir = "./assets/ocr"
    if not os.path.exists(assets_dir):
        print(f"\n⚠ Warning: Assets directory not found at {assets_dir}")
        print("Examples will be skipped. Use example_custom_ocr() with your own images.")
        print("\nTo download sample images:")
        print("1. Create ./assets/ocr directory")
        print("2. Add sample images for OCR testing")
        return
    
    print("\nRunning OCR examples... (This may take a few minutes)\n")
    
    # Example 1: English OCR
    if os.path.exists(f"{assets_dir}/ocr_example2.jpg"):
        try:
            example_1_full_page_ocr_english(f"{assets_dir}/ocr_example2.jpg")
        except Exception as e:
            print(f"\nExample 1 failed: {e}\n")
    
    # Example 2: Multilingual OCR
    if os.path.exists(f"{assets_dir}/ocr_example6.jpg"):
        try:
            example_2_full_page_ocr_multilingual(f"{assets_dir}/ocr_example6.jpg")
        except Exception as e:
            print(f"\nExample 2 failed: {e}\n")
    
    # Example 3: Text spotting (line-level)
    if os.path.exists(f"{assets_dir}/ocr_example3.jpg"):
        try:
            example_3_text_spotting_line_level(f"{assets_dir}/ocr_example3.jpg")
        except Exception as e:
            print(f"\nExample 3 failed: {e}\n")
    
    # Example 4: Text spotting (word-level)
    if os.path.exists(f"{assets_dir}/ocr_example2.jpg"):
        try:
            example_4_text_spotting_word_level(f"{assets_dir}/ocr_example2.jpg")
        except Exception as e:
            print(f"\nExample 4 failed: {e}\n")
    
    # Example 5: Information extraction
    if os.path.exists(f"{assets_dir}/ocr_example3.jpg"):
        try:
            keys = {"company": "", "date": "", "address": "", "total": ""}
            example_5_visual_information_extraction(f"{assets_dir}/ocr_example3.jpg", keys)
        except Exception as e:
            print(f"\nExample 5 failed: {e}\n")
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nTo run custom OCR:")
    print("1. Edit this script")
    print("2. Use example_custom_ocr() function")
    print("3. Provide your own image path and prompt")
    print("\nExample:")
    print("  example_custom_ocr('my_image.jpg', 'Extract all text')")
    print("="*70 + "\n")


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

def print_usage():
    """Print usage instructions"""
    usage = """
═══════════════════════════════════════════════════════════════════════
USAGE INSTRUCTIONS - OCR with Qwen3-VL (Local)
═══════════════════════════════════════════════════════════════════════

This script demonstrates OCR capabilities using a local Qwen3-VL server.

PREREQUISITES:
--------------
1. Local LLM server running at http://localhost:8080/v1
2. Model: Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm (or update LOCAL_MODEL_ID)
3. Model must support vision/image input

RUN THE SCRIPT:
--------------
python ocr_local.py

The script will:
1. Auto-install missing dependencies
2. Test connection to your local server
3. Run OCR examples (if assets directory exists)

FEATURES:
---------
1. Full-page OCR (English and multilingual)
2. Text spotting with bounding boxes (line and word level)
3. Visual information extraction (forms, receipts, invoices)
4. Custom OCR with your own images

EXAMPLES:
---------

1. Basic OCR:
   example_1_full_page_ocr_english("document.jpg")

2. Multilingual OCR:
   example_2_full_page_ocr_multilingual("chinese_doc.jpg")

3. Text spotting (get positions):
   example_3_text_spotting_line_level("receipt.jpg")

4. Extract specific information:
   keys = {"company": "", "date": "", "total": ""}
   example_5_visual_information_extraction("invoice.jpg", keys)

5. Custom OCR:
   example_custom_ocr("my_image.jpg", "Read all text in the image")

ASSETS DIRECTORY:
-----------------
Place sample images in: ./assets/ocr/
Expected files:
  - ocr_example2.jpg (English text)
  - ocr_example3.jpg (Receipt/invoice)
  - ocr_example6.jpg (Multilingual text)

BOUNDING BOX FORMAT:
--------------------
Text spotting returns coordinates in [x1, y1, x2, y2] format:
- Coordinates are relative (0-999 range)
- Converted to absolute pixels for display
- Format: [{'bbox_2d': [x1, y1, x2, y2], 'text_content': 'text'}, ...]

CONFIGURATION:
--------------
Edit at top of script:

LOCAL_MODEL_ID = 'your-model-id'
LOCAL_API_BASE = 'http://your-server:port/v1'

TROUBLESHOOTING:
----------------
1. Server not responding:
   curl http://localhost:8080/v1/models

2. Vision not working:
   - Ensure model supports vision/multimodal input
   - Check server logs for errors

3. Base64 encoding issues:
   - Ensure images are valid JPEG/PNG
   - Check file permissions

4. JSON parsing errors:
   - Model may not support structured output well
   - Try simpler prompts
   - Check raw output for formatting issues

NOTES:
------
- Coordinates use 0-999 relative scale (not 0-1)
- Large images may take longer to process
- Word-level spotting produces more bounding boxes
- Information extraction works best with structured documents

═══════════════════════════════════════════════════════════════════════
"""
    print(usage)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print_usage()
    else:
        main()