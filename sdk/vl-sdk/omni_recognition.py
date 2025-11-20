#!/usr/bin/env python3
# coding: utf-8

"""
Omni Recognition with Qwen3-VL (Local Version)

This script demonstrates Qwen3-VL's omni recognition capabilities using a local LLM server:
- Object recognition (celebrities, anime characters, landmarks, animals, plants)
- Object spotting with bounding boxes
- Multi-object identification

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
LOCAL_MODEL_ID = 'Qwen3-Omni-10k'
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
        if line.strip() in ["```json", "```"]:
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


def display_bounding_boxes_terminal(image_path, bounding_boxes_json):
    """
    Display bounding boxes in terminal format
    
    Args:
        image_path: Path to the image
        bounding_boxes_json: JSON string with bounding box data
    """
    # Load image to get dimensions
    img = Image.open(image_path)
    width, height = img.size
    
    print(f"\nImage dimensions: {width}x{height}")
    print("-" * 70)
    
    # Parse JSON
    parsed_json = parse_json_output(bounding_boxes_json)
    
    try:
        # Try to parse as JSON
        boxes = json.loads(parsed_json)
        
        # Handle different JSON structures
        if isinstance(boxes, dict):
            # If it's a dict, look for common keys
            if 'objects' in boxes:
                boxes = boxes['objects']
            elif 'items' in boxes:
                boxes = boxes['items']
            elif 'results' in boxes:
                boxes = boxes['results']
        
        if not isinstance(boxes, list):
            print("Warning: Expected a list of objects")
            boxes = [boxes]
        
        print(f"\nFound {len(boxes)} objects:\n")
        
        for i, box in enumerate(boxes, 1):
            # Get name/label
            name = box.get('name', box.get('label', box.get('text', 'Unknown')))
            name_en = box.get('name_en', box.get('english_name', ''))
            name_cn = box.get('name_cn', box.get('chinese_name', ''))
            
            # Display names
            if name_en and name_cn:
                print(f"[{i}] {name_cn} ({name_en})")
            elif name_en or name_cn:
                print(f"[{i}] {name_en or name_cn}")
            else:
                print(f"[{i}] {name}")
            
            # Get bounding box if present
            bbox = box.get('bbox', box.get('bounding_box', box.get('bbox_2d', None)))
            
            if bbox:
                # Handle different bbox formats
                if len(bbox) == 4:
                    # Assume [x1, y1, x2, y2] in relative coords (0-1000)
                    abs_x1 = int(bbox[0] / 1000 * width)
                    abs_y1 = int(bbox[1] / 1000 * height)
                    abs_x2 = int(bbox[2] / 1000 * width)
                    abs_y2 = int(bbox[3] / 1000 * height)
                    
                    # Ensure correct order
                    if abs_x1 > abs_x2:
                        abs_x1, abs_x2 = abs_x2, abs_x1
                    if abs_y1 > abs_y2:
                        abs_y1, abs_y2 = abs_y2, abs_y1
                    
                    print(f"    Position: ({abs_x1}, {abs_y1}) → ({abs_x2}, {abs_y2})")
                    print(f"    Size: {abs_x2-abs_x1}x{abs_y2-abs_y1} pixels")
            
            # Display confidence if present
            confidence = box.get('confidence', box.get('score', None))
            if confidence:
                print(f"    Confidence: {confidence:.2%}" if confidence <= 1 else f"    Confidence: {confidence}")
            
            print()
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"\nRaw output:\n{parsed_json}")
    except Exception as e:
        print(f"Error processing bounding boxes: {e}")
        print(f"\nRaw output:\n{bounding_boxes_json}")


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
        print("2. The server is accessible at http://localhost:10010")
        print("3. The /v1/models endpoint is available")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


# ============================================================================
# OMNI RECOGNITION EXAMPLES
# ============================================================================

def example_1_object_recognition(image_path, query="Who is this?"):
    """
    Example 1: Object Recognition
    
    Recognizes celebrities, anime characters, landmarks, animals, plants, etc.
    
    Args:
        image_path: Path to the image
        query: Question about the image
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Object Recognition")
    print("="*70)
    print("\nRecognize celebrities, anime characters, landmarks, animals, plants, etc.\n")
    
    print(f"Image: {image_path}")
    print(f"Query: {query}")
    print("\n" + "-"*70)
    print("RECOGNITION RESULT:")
    print("-"*70 + "\n")
    
    try:
        if not os.path.exists(image_path):
            print(f"✗ Error: Image not found at {image_path}")
            return None
        
        # Display image info
        img = Image.open(image_path)
        print(f"Image size: {img.size[0]}x{img.size[1]} pixels\n")
        
        response = inference_local(image_path, query)
        print(response)
        print("\n" + "="*70 + "\n")
        return response
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def example_2_object_spotting(image_path, query):
    """
    Example 2: Object Spotting with Bounding Boxes
    
    Identifies multiple objects and provides their coordinates.
    
    Args:
        image_path: Path to the image
        query: Question requesting object identification with bounding boxes
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Object Spotting with Bounding Boxes")
    print("="*70)
    print("\nIdentify multiple objects and their locations in the image.\n")
    
    print(f"Image: {image_path}")
    print(f"Query: {query}")
    print("\n" + "-"*70)
    print("SPOTTING RESULT:")
    print("-"*70 + "\n")
    
    try:
        if not os.path.exists(image_path):
            print(f"✗ Error: Image not found at {image_path}")
            return None
        
        # Display image info
        img = Image.open(image_path)
        print(f"Image size: {img.size[0]}x{img.size[1]} pixels\n")
        
        response = inference_local(image_path, query, max_tokens=8192)
        
        print("Raw JSON output:")
        print(response)
        print("\n" + "-"*70)
        
        # Try to parse and display bounding boxes
        try:
            display_bounding_boxes_terminal(image_path, response)
        except Exception as e:
            print(f"Could not parse bounding boxes: {e}")
        
        print("="*70 + "\n")
        return response
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def example_3_multi_object_recognition(image_path):
    """
    Example 3: Multi-object Recognition
    
    Recognizes multiple objects in a complex scene.
    
    Args:
        image_path: Path to the image
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Multi-object Recognition")
    print("="*70)
    
    prompt = "Identify all recognizable objects in this image. List them with their names and descriptions."
    
    print(f"\nImage: {image_path}")
    print(f"Prompt: {prompt}")
    print("\n" + "-"*70)
    print("RECOGNITION RESULT:")
    print("-"*70 + "\n")
    
    try:
        if not os.path.exists(image_path):
            print(f"✗ Error: Image not found at {image_path}")
            return None
        
        # Display image info
        img = Image.open(image_path)
        print(f"Image size: {img.size[0]}x{img.size[1]} pixels\n")
        
        response = inference_local(image_path, prompt, max_tokens=6144)
        print(response)
        print("\n" + "="*70 + "\n")
        return response
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def example_custom_recognition(image_path, query):
    """
    Custom recognition with your own image and query
    
    Args:
        image_path: Path to your image
        query: Your question about the image
    """
    print("\n" + "="*70)
    print("CUSTOM OMNI RECOGNITION")
    print("="*70)
    
    print(f"\nImage: {image_path}")
    print(f"Query: {query}")
    print("\n" + "-"*70)
    print("RESULT:")
    print("-"*70 + "\n")
    
    try:
        if not os.path.exists(image_path):
            print(f"✗ Error: Image not found at {image_path}")
            return None
        
        # Display image info
        img = Image.open(image_path)
        print(f"Image size: {img.size[0]}x{img.size[1]} pixels\n")
        
        response = inference_local(image_path, query, max_tokens=8192)
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
    print("Omni Recognition with Qwen3-VL - Local Version")
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
    print("Ready to run omni recognition examples!")
    print("="*70)
    
    # Check if assets directory exists
    assets_dir = "./assets/omni_recognition"
    if not os.path.exists(assets_dir):
        print(f"\n⚠ Warning: Assets directory not found at {assets_dir}")
        print("Examples will be skipped. Use example_custom_recognition() with your own images.")
        print("\nTo use sample images:")
        print("1. Create ./assets/omni_recognition directory")
        print("2. Add sample images for recognition testing")
        return
    
    print("\nRunning omni recognition examples... (This may take a few minutes)\n")
    
    # Example 1: Celebrity recognition
    if os.path.exists(f"{assets_dir}/sample-celebrity-2.jpg"):
        try:
            example_1_object_recognition(
                f"{assets_dir}/sample-celebrity-2.jpg",
                "Who is this person?"
            )
        except Exception as e:
            print(f"\nExample 1 failed: {e}\n")
    
    # Example 2: Food spotting with bounding boxes
    if os.path.exists(f"{assets_dir}/sample-food.jpeg"):
        try:
            example_2_object_spotting(
                f"{assets_dir}/sample-food.jpeg",
                "Identify food in the image and return their bounding box and Chinese and English name in JSON format."
            )
        except Exception as e:
            print(f"\nExample 2 failed: {e}\n")
    
    # Example 3: Anime character spotting
    if os.path.exists(f"{assets_dir}/sample-anime.jpeg"):
        try:
            example_2_object_spotting(
                f"{assets_dir}/sample-anime.jpeg",
                "Who are the anime characters in the image? Please show the bounding boxes of all characters and their names in Chinese and English in JSON format."
            )
        except Exception as e:
            print(f"\nExample 3 failed: {e}\n")
    
    # Example 4: Bird recognition
    if os.path.exists(f"{assets_dir}/sample-bird.jpg"):
        try:
            example_1_object_recognition(
                f"{assets_dir}/sample-bird.jpg",
                "What kind of bird is this? Provide details about the species."
            )
        except Exception as e:
            print(f"\nExample 4 failed: {e}\n")
    
    # Example 5: General celebrity recognition
    if os.path.exists(f"{assets_dir}/sample-celebrity.jpeg"):
        try:
            example_1_object_recognition(
                f"{assets_dir}/sample-celebrity.jpeg",
                "Who is this celebrity? Provide information about them."
            )
        except Exception as e:
            print(f"\nExample 5 failed: {e}\n")
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nTo run custom recognition:")
    print("1. Edit this script")
    print("2. Use example_custom_recognition() function")
    print("3. Provide your own image path and query")
    print("\nExample:")
    print("  example_custom_recognition('my_image.jpg', 'What is this?')")
    print("="*70 + "\n")


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

def print_usage():
    """Print usage instructions"""
    usage = """
═══════════════════════════════════════════════════════════════════════
USAGE INSTRUCTIONS - Omni Recognition with Qwen3-VL (Local)
═══════════════════════════════════════════════════════════════════════

This script demonstrates omni recognition capabilities using local Qwen3-VL.

PREREQUISITES:
--------------
1. Local LLM server running at http://localhost:10010/v1
2. Model: Qwen3-Omni-10k loaded on the server
3. Model must support vision/image input

RUN THE SCRIPT:
--------------
python omni_recognition_local.py

The script will:
1. Auto-install missing dependencies
2. Test connection to your local server
3. Run recognition examples (if assets directory exists)

FEATURES:
---------
1. Object Recognition - Identify celebrities, anime characters, landmarks
2. Object Spotting - Detect multiple objects with bounding boxes
3. Multi-object Recognition - Comprehensive scene analysis
4. Custom Recognition - Use your own images and queries

EXAMPLES:
---------

1. Celebrity Recognition:
   example_1_object_recognition("celebrity.jpg", "Who is this?")

2. Anime Character Spotting:
   example_2_object_spotting(
       "anime.jpg",
       "Who are the characters? Show bounding boxes in JSON."
   )

3. Food Recognition:
   example_2_object_spotting(
       "food.jpg",
       "Identify food items with bounding boxes in JSON."
   )

4. Animal/Plant Recognition:
   example_1_object_recognition("bird.jpg", "What species is this?")

5. Custom Query:
   example_custom_recognition("my_image.jpg", "What is this object?")

ASSETS DIRECTORY:
-----------------
Place sample images in: ./assets/omni_recognition/
Expected files:
  - sample-celebrity.jpeg
  - sample-celebrity-2.jpg
  - sample-anime.jpeg
  - sample-food.jpeg
  - sample-bird.jpg

BOUNDING BOX FORMAT:
--------------------
Object spotting returns JSON with coordinates:
- Format: [x1, y1, x2, y2] in relative scale (0-1000)
- Converted to absolute pixels for display
- Includes names in both Chinese and English (when applicable)

Example JSON structure:
[
  {
    "name_en": "Rice Bowl",
    "name_cn": "米饭",
    "bbox": [100, 200, 300, 400]
  }
]

CONFIGURATION:
--------------
Edit at top of script:

LOCAL_MODEL_ID = 'Qwen3-Omni-10k'
LOCAL_API_BASE = 'http://localhost:10010/v1'

RECOGNITION CAPABILITIES:
-------------------------
- Celebrities and public figures
- Anime and manga characters
- Food items and dishes
- Animals and plants (species identification)
- Landmarks and buildings
- Products and brands
- Vehicles and objects
- Text and signs

TROUBLESHOOTING:
----------------
1. Server not responding:
   curl http://localhost:10010/v1/models

2. Vision not working:
   - Ensure Qwen3-Omni model is loaded
   - Check server logs for errors
   - Verify --allowed-local-media-path is set

3. Bounding boxes incorrect:
   - Model uses relative coordinates (0-1000)
   - Verify image dimensions
   - Check JSON parsing

4. Recognition accuracy:
   - Use clear, high-quality images
   - Provide specific queries
   - Try different prompt phrasings

5. Connection errors:
   - Verify port 10010 is correct
   - Check if model is still loading
   - Review server startup logs

NOTES:
------
- Best results with clear, well-lit images
- Supports multiple objects in one image
- Can identify obscure anime characters and celebrities
- Works with various languages (Chinese, English, etc.)
- Bounding boxes use 0-1000 relative coordinate system

═══════════════════════════════════════════════════════════════════════
"""
    print(usage)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print_usage()
    else:
        main()