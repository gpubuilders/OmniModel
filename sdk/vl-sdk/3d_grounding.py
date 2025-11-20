#!/usr/bin/env python
# coding: utf-8

# ### 3D Grounding with Qwen3-VL (Local Version)
#
# This notebook showcases Qwen3-VL's advanced spatial understanding capabilities, including 3D object localization and specific target grounding within images.
# This version has been modified to use a local LLM server instead of DashScope API.
#
# See how it integrates visual and linguistic understanding to interpret complex 3D scenes effectively.

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
    "opencv-python",
    "numpy",
    "matplotlib",
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


# @title 3D Visualization Utilities

import json
import random
import io
import ast
import math
import cv2
import numpy as np
from PIL import ImageColor
import matplotlib.pyplot as plt
import base64
from PIL import Image


def parse_bbox_3d_from_text(text: str) -> list:
    """
    Parse 3D bounding box information from assistant response.

    Args:
        text: Assistant response text containing JSON with bbox_3d information

    Returns:
        List of dictionaries containing bbox_3d data
    """
    try:
        # Find JSON content
        if "```json" in text:
            start_idx = text.find("```json")
            end_idx = text.find("```", start_idx + 7)
            if end_idx != -1:
                json_str = text[start_idx + 7:end_idx].strip()
            else:
                json_str = text[start_idx + 7:].strip()
        else:
            # Find first [ and last ]
            start_idx = text.find('[')
            end_idx = text.rfind(']')
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx + 1]
            else:
                return []

        # Parse JSON
        bbox_data = json.loads(json_str)

        # Normalize to list format
        if isinstance(bbox_data, list):
            return bbox_data
        elif isinstance(bbox_data, dict):
            return [bbox_data]
        else:
            return []

    except (json.JSONDecodeError, IndexError, KeyError):
        return []


def convert_3dbbox(point, cam_params):
    """Convert 3D bounding box to 2D image coordinates"""
    x, y, z, x_size, y_size, z_size, pitch, yaw, roll = point
    hx, hy, hz = x_size / 2, y_size / 2, z_size / 2
    local_corners = [
        [ hx,  hy,  hz],
        [ hx,  hy, -hz],
        [ hx, -hy,  hz],
        [ hx, -hy, -hz],
        [-hx,  hy,  hz],
        [-hx,  hy, -hz],
        [-hx, -hy,  hz],
        [-hx, -hy, -hz]
    ]

    def rotate_xyz(_point, _pitch, _yaw, _roll):
        x0, y0, z0 = _point
        x1 = x0
        y1 = y0 * math.cos(_pitch) - z0 * math.sin(_pitch)
        z1 = y0 * math.sin(_pitch) + z0 * math.cos(_pitch)

        x2 = x1 * math.cos(_yaw) + z1 * math.sin(_yaw)
        y2 = y1
        z2 = -x1 * math.sin(_yaw) + z1 * math.cos(_yaw)

        x3 = x2 * math.cos(_roll) - y2 * math.sin(_roll)
        y3 = x2 * math.sin(_roll) + y2 * math.cos(_roll)
        z3 = z2

        return [x3, y3, z3]

    img_corners = []
    for corner in local_corners:
        rotated = rotate_xyz(corner, np.deg2rad(pitch), np.deg2rad(yaw), np.deg2rad(roll))
        X, Y, Z = rotated[0] + x, rotated[1] + y, rotated[2] + z
        if Z > 0:
            x_2d = cam_params['fx'] * (X / Z) + cam_params['cx']
            y_2d = cam_params['fy'] * (Y / Z) + cam_params['cy']
            img_corners.append([x_2d, y_2d])

    return img_corners


def draw_3dbboxes(image_path, cam_params, bbox_3d_list, color=None):
    """Draw multiple 3D bounding boxes on the same image and return matplotlib figure"""
    # Read image
    annotated_image = cv2.imread(image_path)
    if annotated_image is None:
        print(f"Error reading image: {image_path}")
        return None

    edges = [
        [0,1], [2,3], [4,5], [6,7],
        [0,2], [1,3], [4,6], [5,7],
        [0,4], [1,5], [2,6], [3,7]
    ]

    # Draw 3D box for each bbox
    for bbox_data in bbox_3d_list:
        # Extract bbox_3d from the dictionary
        if isinstance(bbox_data, dict) and 'bbox_3d' in bbox_data:
            bbox_3d = bbox_data['bbox_3d']
        else:
            bbox_3d = bbox_data

        # Convert angles multiplied by 180 to degrees
        bbox_3d = list(bbox_3d)  # Convert to list for modification
        bbox_3d[-3:] = [_x * 180 for _x in bbox_3d[-3:]]
        bbox_2d = convert_3dbbox(bbox_3d, cam_params)

        if len(bbox_2d) >= 8:
            # Generate random color for each box
            box_color = [random.randint(0, 255) for _ in range(3)]
            for start, end in edges:
                try:
                    pt1 = tuple([int(_pt) for _pt in bbox_2d[start]])
                    pt2 = tuple([int(_pt) for _pt in bbox_2d[end]])
                    cv2.line(annotated_image, pt1, pt2, box_color, 2)
                except:
                    continue

    # Convert BGR to RGB for matplotlib
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Create matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(annotated_image_rgb)
    ax.axis('off')

    return fig


# ## 3D Object Localization with Qwen3-VL
#
# Qwen3-VL supports localizing specific 3D objects in images based on natural language descriptions. This notebook demonstrates various 3D grounding scenarios.
#
# Because accurate 3D perception highly relies on camera parameters, please make sure you have camera intrinsic parameters (focal length fx, fy and principal point cx, cy) for better experience. If you don't have camera parameters, we will generate a group of general camera parameters with fov=60° for you to try this demo.
#
# ### 3D Bounding Box Format
#
# We represent 3D bounding boxes as: `[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]`
#
# - **x_center, y_center, z_center**: Object center in camera coordinates (meters)
# - **x_size, y_size, z_size**: Object dimensions (meters)
# - **roll, pitch, yaw**: Rotation angles (radians)
#

# API Inference function using local OpenAI-compatible server

def encode_image(image_path):
    """Encode image to base64 format"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def inference_with_local_api(image_path, prompt, model_id=LOCAL_MODEL_ID):
    """
    Perform inference using local OpenAI-compatible API server.

    Args:
        image_path: Path to image input (local file path)
        prompt: Text prompt for the model
        model_id: Model ID to use for inference

    Returns:
        str: Generated text response from the model
    """
    base64_image = encode_image(image_path)
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_HTTP_API_URL'),
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error during API call: {e}")
        raise


# ### Camera Parameters Generation
#
# Note: When you don't have access to the original camera intrinsic parameters, we can generate general camera parameters with a field of view of 60° for demonstration purposes.
#

def load_camera_params(image_name):
    """Load camera parameters for a specific image from cam_infos.json"""
    try:
        with open('./assets/spatial_understanding/cam_infos.json', 'r') as f:
            cam_infos = json.load(f)
        return cam_infos.get(image_name, None)
    except FileNotFoundError:
        print(f"Camera info file not found. Using default parameters for {image_name}")
        return None

def generate_camera_params(image_path, fx=None, fy=None, cx=None, cy=None, fov=60):
    """
    Generate camera parameters for 3D visualization.

    Args:
        image_path: Path to the image
        fx, fy: Focal lengths in pixels (if None, will be calculated from fov)
        cx, cy: Principal point coordinates in pixels (if None, will be set to image center)
        fov: Field of view in degrees (default: 60°)

    Returns:
        dict: Camera parameters with keys 'fx', 'fy', 'cx', 'cy'
    """
    image = Image.open(image_path)
    w, h = image.size

    # Generate pseudo camera params if not provided
    if fx is None or fy is None:
        fx = round(w / (2 * np.tan(np.deg2rad(fov) / 2)), 2)
        fy = round(h / (2 * np.tan(np.deg2rad(fov) / 2)), 2)

    if cx is None or cy is None:
        cx = round(w / 2, 2)
        cy = round(h / 2, 2)

    cam_params = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    return cam_params


# Example 1: Detect all cars in autonomous driving scene
def example_1_detect_cars():
    """Example 1: Detect all cars in autonomous driving scene"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Detect All Cars in Autonomous Driving Scene")
    print("="*70)

    image_path = "./assets/spatial_understanding/autonomous_driving.jpg"
    prompt = "Find all cars in this image. For each car, provide its 3D bounding box. The output format required is JSON: `[{\"bbox_3d\":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],\"label\":\"category\"}]`."

    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    
    if os.path.exists(image_path):
        # Load camera parameters
        cam_params = load_camera_params("autonomous_driving.jpg")
        if not cam_params:
            cam_params = generate_camera_params(image_path, fov=60)
        
        # Call local API to get 3D bounding box results
        response = inference_with_local_api(image_path, prompt)
        print("\nResponse:", response)
        
        bbox_3d_results = parse_bbox_3d_from_text(response)
        print("Parsed bbox_3d_results:", bbox_3d_results)
        
        # Display the 3D bounding boxes visualization
        if len(bbox_3d_results) > 0:
            fig = draw_3dbboxes(image_path, cam_params, bbox_3d_results)
            if fig is not None:
                plt.show()
    else:
        print(f"Image file {image_path} does not exist. Please download sample images to run this example.")
    
    print("="*70 + "\n")


# Example 2: Detect a specific object using descriptions
def example_2_detect_specific_object():
    """Example 2: Detect a specific object using descriptions"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Detect Specific Object Using Descriptions")
    print("="*70)

    image_path = "./assets/spatial_understanding/office.jpg"
    prompt = "Locate the black chair in image and provide 3D bounding boxes results in JSON format."

    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    
    if os.path.exists(image_path):
        # Load camera parameters
        cam_params = load_camera_params("office.jpg")
        if not cam_params:
            cam_params = generate_camera_params(image_path, fov=60)
        
        # Call local API to get 3D bounding box results
        response = inference_with_local_api(image_path, prompt)
        print("\nResponse:", response)
        
        bbox_3d_results = parse_bbox_3d_from_text(response)
        print("Parsed bbox_3d_results:", bbox_3d_results)
        
        # Display the 3D bounding boxes visualization
        if len(bbox_3d_results) > 0:
            fig = draw_3dbboxes(image_path, cam_params, bbox_3d_results)
            if fig is not None:
                plt.show()
    else:
        print(f"Image file {image_path} does not exist. Please download sample images to run this example.")
    
    print("="*70 + "\n")


# Example 3: Detect multiple objects simultaneously
def example_3_detect_multiple_objects():
    """Example 3: Detect multiple objects simultaneously"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Detect Multiple Objects Simultaneously")
    print("="*70)

    image_path = "./assets/spatial_understanding/lounge.jpg"
    prompt = "在提供的图像里定位桌子、椅子和沙发，输出对应的三维边界框。格式为：[{\"bbox_3d\":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],\"label\":\"类别\"}]。"

    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    
    if os.path.exists(image_path):
        # Load camera parameters
        cam_params = load_camera_params("lounge.jpg")
        if not cam_params:
            cam_params = generate_camera_params(image_path, fov=60)
        
        # Call local API to get 3D bounding box results
        response = inference_with_local_api(image_path, prompt)
        print("\nResponse:", response)
        
        bbox_3d_results = parse_bbox_3d_from_text(response)
        print("Parsed bbox_3d_results:", bbox_3d_results)
        
        # Display the 3D bounding boxes visualization
        if len(bbox_3d_results) > 0:
            fig = draw_3dbboxes(image_path, cam_params, bbox_3d_results)
            if fig is not None:
                plt.show()
    else:
        print(f"Image file {image_path} does not exist. Please download sample images to run this example.")
    
    print("="*70 + "\n")


# Example 4: Using custom camera parameters
def example_4_custom_camera_params():
    """Example 4: Using custom camera parameters"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Using Custom Camera Parameters")
    print("="*70)

    image_path = "./assets/spatial_understanding/manipulation.jpg"
    prompt = "Detect the bottle in the image and predict the 3D box. Output JSON: [{\"bbox_3d\":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],\"label\":\"category\"}]."

    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    
    if os.path.exists(image_path):
        # Generate camera parameters using our function (since we don't have original camera params)
        cam_params = generate_camera_params(image_path, fov=60)
        
        # Call local API to get 3D bounding box results
        response = inference_with_local_api(image_path, prompt)
        print("\nResponse:", response)
        
        bbox_3d_results = parse_bbox_3d_from_text(response)
        print("Parsed bbox_3d_results:", bbox_3d_results)
        
        # Display the 3D bounding boxes visualization
        if len(bbox_3d_results) > 0:
            fig = draw_3dbboxes(image_path, cam_params, bbox_3d_results)
            if fig is not None:
                plt.show()
    else:
        print(f"Image file {image_path} does not exist. Please download sample images to run this example.")
    
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
    print("3D Grounding with Local LLM")
    print("="*60)
    print(f"Model: {LOCAL_MODEL_ID}")
    print(f"Endpoint: {os.getenv('OPENAI_BASE_HTTP_API_URL')}")
    print("="*60)

    # Test API connection first
    if not test_api_connection():
        print("\nCannot proceed without API connection.")
        return

    print("\nTo run 3D grounding examples, make sure you have the required assets in ./assets/spatial_understanding/")
    
    # Run examples if assets exist
    print("\nRunning 3D grounding examples...")
    
    # Example 1: Detect cars
    example_1_detect_cars()
    
    # Example 2: Detect specific object
    example_2_detect_specific_object()
    
    # Example 3: Detect multiple objects
    example_3_detect_multiple_objects()
    
    # Example 4: Custom camera parameters
    example_4_custom_camera_params()

    print("Done!")


if __name__ == "__main__":
    main()