#!/usr/bin/env python
# coding: utf-8

# ### Computer Use with Qwen3-VL (Local Version)
#
# This notebook demonstrates how to use Qwen3-VL for computer use. It takes a screenshot of a user's desktop and a query, and then uses the model to interpret the user's query on the screenshot.
# This version has been modified to use a local LLM server instead of DashScope API.

import os
import sys
import subprocess
from openai import OpenAI
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Local LLM Server Configuration
os.environ['OPENAI_API_KEY'] = 'dummy-key'  # Local server might not validate this
os.environ['OPENAI_BASE_HTTP_API_URL'] = 'http://localhost:8080/v1'
LOCAL_MODEL_ID = 'Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm'

# Detect if running in Jupyter or terminal
try:
    get_ipython()
    IN_JUPYTER = True
    from IPython.display import Markdown, display as jupyter_display
except NameError:
    IN_JUPYTER = False

def display(content):
    """Display content - works in both Jupyter and terminal"""
    if IN_JUPYTER:
        from IPython.display import display as jupyter_display
        jupyter_display(content)
    else:
        if hasattr(content, 'save'):  # PIL Image
            print("[Image would be displayed here in Jupyter]")
            print(f"Image size: {content.size}")
        else:
            print(content)

def Markdown(text):
    """Markdown wrapper - works in both Jupyter and terminal"""
    if IN_JUPYTER:
        from IPython.display import Markdown as JupyterMarkdown
        return JupyterMarkdown(text)
    else:
        return text

def install_package(package):
    """Install a package if it's not already installed"""
    try:
        __import__(package.split('[')[0])
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", package])
        print(f"✓ {package} installed successfully")

# Check and install required packages
print("="*70)
print("Checking and installing dependencies...")
print("="*70)

required_packages = [
    "Pillow",
    "requests",
    "openai",
    "qwen-agent",
]

for package in required_packages:
    install_package(package)

print("\n✓ All dependencies installed!\n")

# Optional: Only uncomment if you want to run local inference (not API-based)
# from transformers import AutoProcessor, AutoModelForVision2Seq
# from qwen_vl_utils import process_vision_info
# model_path = "path/to/local/model"
# processor = AutoProcessor.from_pretrained(model_path)
# model, output_loading_info = AutoModelForVision2Seq.from_pretrained(
#     model_path,
#     torch_dtype="auto",
#     device_map="auto",
#     output_loading_info=True
# )
# print("output_loading_info", output_loading_info)


from PIL import Image, ImageDraw, ImageColor

def draw_point(image: Image.Image, point: list, color=None):
    if isinstance(color, str):
        try:
            color = ImageColor.getrgb(color)
            color = color + (128,)
        except ValueError:
            color = (255, 0, 0, 128)
    else:
        color = (255, 0, 0, 128)

    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    radius = min(image.size) * 0.05
    x, y = point

    overlay_draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=color
    )

    center_radius = radius * 0.1
    overlay_draw.ellipse(
        [(x - center_radius, y - center_radius),
         (x + center_radius, y + center_radius)],
        fill=(0, 255, 0, 255)
    )

    image = image.convert('RGBA')
    combined = Image.alpha_composite(image, overlay)

    return combined.convert('RGB')


# API Inference function using local OpenAI-compatible server

import json
import base64
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize

from utils.agent_function_call import ComputerUse

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def perform_gui_grounding_with_local_api(screenshot_path, user_query, model_id=LOCAL_MODEL_ID, min_pixels=3136, max_pixels=12845056):
    """
    Perform GUI grounding using local Qwen model to interpret user query on a screenshot.

    Args:
        screenshot_path (str): Path to the screenshot image
        user_query (str): User's query/instruction
        model_id: Model ID to use for inference
        min_pixels: Minimum pixels for the image
        max_pixels: Maximum pixels for the image

    Returns:
        tuple: (output_text, display_image) - Model's output text and annotated image
    """

    # Open and process image
    input_image = Image.open(screenshot_path)
    base64_image = encode_image(screenshot_path)
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_HTTP_API_URL'),
    )
    resized_height, resized_width = smart_resize(
        input_image.height,
        input_image.width,
        factor=32,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # Initialize computer use function
    computer_use = ComputerUse(
        cfg={"display_width_px": 1000, "display_height_px": 1000}
    )

    # Build messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                    # PNG image:  f"data:image/png;base64,{base64_image}"
                    # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                    # WEBP image: f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": user_query},
            ],
        }
    ]
    print(json.dumps(messages, indent=4))
    
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
    )

    output_text = completion.choices[0].message.content

    # Parse action and visualize - need to handle cases where parsing might fail
    try:
        # Extract the coordinate from the response - need to handle different possible formats
        # Look for the coordinate in the response
        import re
        
        # Try to find coordinate values in the response
        x_match = re.search(r'"x"\s*:\s*(\d+\.?\d*)', output_text)
        y_match = re.search(r'"y"\s*:\s*(\d+\.?\d*)', output_text)
        
        if x_match and y_match:
            coord_x = float(x_match.group(1))
            coord_y = float(y_match.group(1))
        else:
            # Look for array format [x, y] 
            coord_matches = re.findall(r'\d+\.?\d*', output_text[-200:])  # Look in last 200 chars
            if len(coord_matches) >= 2:
                coord_x = float(coord_matches[-2])
                coord_y = float(coord_matches[-1])
            else:
                # Default to center if we can't parse
                coord_x = 500
                coord_y = 500
        
        # Convert relative coordinates to absolute
        display_x = coord_x / 1000 * resized_width
        display_y = coord_y / 1000 * resized_height
        
        display_image = input_image.resize((resized_width, resized_height))
        display_image = draw_point(display_image, (display_x, display_y), color='green')

        return output_text, display_image

    except Exception as e:
        print(f"Error parsing action or creating display image: {e}")
        # Return the text and original image if parsing fails
        display_image = input_image.resize((resized_width, resized_height))
        return output_text, display_image


# Example usage functions
def example_1_open_third_issue():
    """Example 1: Open the third issue"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Open the Third Issue")
    print("="*70)

    screenshot = "assets/computer_use/computer_use2.jpeg"
    user_query = 'open the third issue'
    
    print(f"Screenshot: {screenshot}")
    print(f"User query: {user_query}")
    
    if os.path.exists(screenshot):
        try:
            output_text, display_image = perform_gui_grounding_with_local_api(screenshot, user_query, LOCAL_MODEL_ID)
            print("\nOutput text:", output_text)
            display(display_image)
        except Exception as e:
            print(f"Error in example: {e}")
    else:
        print(f"Screenshot file {screenshot} does not exist. Please download sample images to run this example.")
    
    print("="*70 + "\n")


def example_2_reload_cache():
    """Example 2: Reload cache"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Reload Cache")
    print("="*70)

    screenshot = "assets/computer_use/computer_use1.jpeg"
    user_query = 'Reload cache'
    
    print(f"Screenshot: {screenshot}")
    print(f"User query: {user_query}")
    
    if os.path.exists(screenshot):
        try:
            output_text, display_image = perform_gui_grounding_with_local_api(screenshot, user_query, LOCAL_MODEL_ID)
            print("\nOutput text:", output_text)
            display(display_image)
        except Exception as e:
            print(f"Error in example: {e}")
    else:
        print(f"Screenshot file {screenshot} does not exist. Please download sample images to run this example.")
    
    print("="*70 + "\n")


def example_3_open_first_issue():
    """Example 3: Open the first issue"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Open the First Issue")
    print("="*70)

    screenshot = "assets/computer_use/computer_use2.jpeg"
    user_query = 'open the first issue'
    
    print(f"Screenshot: {screenshot}")
    print(f"User query: {user_query}")
    
    if os.path.exists(screenshot):
        try:
            output_text, display_image = perform_gui_grounding_with_local_api(screenshot, user_query, LOCAL_MODEL_ID)
            print("\nOutput text:", output_text)
            display(display_image)
        except Exception as e:
            print(f"Error in example: {e}")
    else:
        print(f"Screenshot file {screenshot} does not exist. Please download sample images to run this example.")
    
    print("="*70 + "\n")


# Simple test function to verify API connection
def test_api_connection():
    """Test if the local API is accessible"""
    print("\n" + "="*60)
    print("Testing API Connection...")
    print("="*60)

    try:
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_HTTP_API_URL'),
        )

        # Try a simple text completion first
        test_messages = [
            {
                "role": "user",
                "content": "Hello! Please respond with 'API connection successful!'"
            }
        ]

        completion = client.chat.completions.create(
            model=LOCAL_MODEL_ID,
            messages=test_messages,
            max_tokens=50
        )

        response = completion.choices[0].message.content
        print(f"\n✓ API Connection Successful!")
        print(f"Model: {LOCAL_MODEL_ID}")
        print(f"Response: {response}\n")
        return True

    except Exception as e:
        print(f"\n✗ API Connection Failed!")
        print(f"Error: {e}\n")
        print("Please check:")
        print("  1. Is your local LLM server running at http://localhost:8080?")
        print("  2. Is the model ID correct: Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm")
        print("  3. Try: curl http://localhost:8080/v1/models\n")
        return False


# Main execution function
def main():
    """
    Main function to run examples.
    """
    print("="*60)
    print("Computer Use with Local LLM")
    print("="*60)
    print(f"Model: {LOCAL_MODEL_ID}")
    print(f"Endpoint: {os.getenv('OPENAI_BASE_HTTP_API_URL')}")
    print("="*60)

    # Test API connection first
    if not test_api_connection():
        print("\nCannot proceed without API connection.")
        return

    print("\nTo run computer use examples, make sure you have the required assets in ./assets/computer_use/")
    
    # Run examples if assets exist
    print("\nRunning computer use examples...")
    
    # Example 1: Open third issue
    example_1_open_third_issue()
    
    # Example 2: Reload cache
    example_2_reload_cache()
    
    # Example 3: Open first issue
    example_3_open_first_issue()

    print("Done!")


if __name__ == "__main__":
    main()