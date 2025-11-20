#!/usr/bin/env python
# coding: utf-8

# ### Spatial Understanding with Qwen3-VL (Local Version)
#
# This notebook showcases Qwen3-VL's advanced spatial localization abilities, including accurate object detection, specific target grounding within images.
# This version has been modified to use a local LLM server instead of DashScope API.
#
# First of all, we list the major updates of Qwen3-VL's spatial understanding abilities as follows:
# * Coordinate System: Qwen3-VL's default coordinate system has been changed from the absolute coordinates used in Qwen2.5-VL to relative coordinates ranging from 0 to 1000. (You don't need to calculate the resized_w)
# * Multi-Target Grounding: Qwen3-VL has improved its multi-target grounding ability.
#
# Now, Let's see how it integrates visual and linguistic understanding to interpret complex scenes effectively.

# #### [Setup]
# Configure local LLM server connection

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


# @title Plotting Util

import json
import random
import io
import ast
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import xml.etree.ElementTree as ET

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def decode_json_points(text: str):
    """Parse coordinate points from text format"""
    try:
        # 清理markdown标记
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]

        # 解析JSON
        data = json.loads(text)
        points = []
        labels = []

        for item in data:
            if "point_2d" in item:
                x, y = item["point_2d"]
                points.append([x, y])

                # 获取label，如果没有则使用默认值
                label = item.get("label", f"point_{len(points)}")
                labels.append(label)

        return points, labels

    except Exception as e:
        print(f"Error: {e}")
        return [], []


def plot_bounding_boxes(im, bounding_boxes):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    except:
        # Fallback to default font if the Noto font is not available
        font = ImageFont.load_default()

    try:
      json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
      end_idx = bounding_boxes.rfind('"}') + len('"}')
      truncated_text = bounding_boxes[:end_idx] + "]"
      json_output = ast.literal_eval(truncated_text)

    if not isinstance(json_output, list):
      json_output = [json_output]

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
      # Select a color from the list
      color = colors[i % len(colors)]

      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int(bounding_box["bbox_2d"][1] / 1000 * height)
      abs_x1 = int(bounding_box["bbox_2d"][0] / 1000 * width)
      abs_y2 = int(bounding_box["bbox_2d"][3] / 1000 * height)
      abs_x2 = int(bounding_box["bbox_2d"][2] / 1000 * width)

      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3
      )

      # Draw the text
      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    # Display the image
    display(img)


def plot_points(im, text):
  img = im
  width, height = img.size
  draw = ImageDraw.Draw(img)
  colors = [
    'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
    'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
    'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
  ] + additional_colors

  points, descriptions = decode_json_points(text)
  print("Parsed points: ", points)
  print("Parsed descriptions: ", descriptions)
  if points is None or len(points) == 0:
    display(img)
    return

  try:
      font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
  except:
      # Fallback to default font if the Noto font is not available
      font = ImageFont.load_default()

  for i, point in enumerate(points):
    color = colors[i % len(colors)]
    abs_x1 = int(point[0])/1000 * width
    abs_y1 = int(point[1])/1000 * height
    radius = 2
    draw.ellipse([(abs_x1 - radius, abs_y1 - radius), (abs_x1 + radius, abs_y1 + radius)], fill=color)
    draw.text((abs_x1 - 20, abs_y1 + 6), descriptions[i], fill=color, font=font)

  display(img)

def plot_points_json(im, text):
  img = im
  width, height = img.size
  draw = ImageDraw.Draw(img)
  colors = [
    'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
    'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
    'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
  ] + additional_colors
  
  try:
      font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
  except:
      # Fallback to default font if the Noto font is not available
      font = ImageFont.load_default()

  text = text.replace('```json', '')
  text = text.replace('```', '')
  data = json.loads(text)
  for item in data:
    point_2d = item['point_2d']
    label = item['label']
    x, y = int(point_2d[0] / 1000 * width), int(point_2d[1] / 1000 * height)
    radius = 2
    draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=colors[0])
    draw.text((x + 2*radius, y + 2*radius), label, fill=colors[0], font=font)

  display(img)


# @title Parsing JSON output
def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


# API Inference function using local OpenAI-compatible server

def inference_with_local_api(img_url, prompt, min_pixels=64 * 32 * 32, max_pixels=9800* 32 * 32):
    """
    Perform inference using local OpenAI-compatible API server.

    Args:
        img_url: Image input (URL string or local file path)
        prompt: Text prompt for the model
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing

    Returns:
        str: Generated text response from the model
    """
    import base64
    import os
    if os.path.exists(img_url):
        with open(img_url, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_image}"
    elif img_url.startswith("http://") or img_url.startswith("https://"):
        image_url = img_url  # For direct URLs
    else:
        raise ValueError("Invalid image URL or file path")

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
                    "image_url": {
                        "url": image_url
                    },
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        completion = client.chat.completions.create(
            model=LOCAL_MODEL_ID,  # Use local model ID instead of qwen3-vl-235b-a22b-instruct
            messages=messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error during API call: {e}")
        raise


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


# Example 1: Detecting different objects on a dining table
def example_1_multi_target_detection():
    """Example 1: Multi-target object detection on a dining table"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Multi-Target Object Detection (Dining Table)")
    print("="*70)

    prompt = 'locate every instance that belongs to the following categories: "plate/dish, scallop, wine bottle, tv, bowl, spoon, air conditioner, coconut drink, cup, chopsticks, person". Report bbox coordinates in JSON format.'
    img_url = "./assets/spatial_understanding/dining_table.png"
    
    print(f"Image: {img_url}")
    print(f"Prompt: {prompt}")
    
    if os.path.exists(img_url):
        model_response = inference_with_local_api(img_url, prompt)
        print("\nResponse:", model_response)
        
        response = Image.open(img_url)
        response.thumbnail([640,640], Image.Resampling.LANCZOS)
        plot_bounding_boxes(response, model_response)
    else:
        print(f"Image file {img_url} does not exist. Please download sample images to run this example.")
    
    print("="*70 + "\n")


# Example 2: Detecting different objects in crowded scenes
def example_2_crowded_scene():
    """Example 2: Detecting objects in crowded scenes"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Detecting Objects in Crowded Scenes")
    print("="*70)

    prompt = 'Locate every instance that belongs to the following categories: "head, hand, man, woman, glasses". Report bbox coordinates in JSON format.'
    img_url = "./assets/spatial_understanding/lots_of_people.jpeg"
    
    print(f"Image: {img_url}")
    print(f"Prompt: {prompt}")
    
    if os.path.exists(img_url):
        model_response = inference_with_local_api(img_url, prompt)
        print("\nResponse:", model_response)
        
        response = Image.open(img_url)
        response.thumbnail([640,640], Image.Resampling.LANCZOS)
        plot_bounding_boxes(response, model_response)
    else:
        print(f"Image file {img_url} does not exist. Please download sample images to run this example.")
    
    print("="*70 + "\n")


# Example 3: Detecting different objects in a 4K drone-view image
def example_3_drone_view():
    """Example 3: Detecting objects in drone view image"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Detecting Objects in Drone View Image")
    print("="*70)

    prompt = 'Locate every instance that belongs to the following categories: "car, bus, bicycle, pedestrian". Report bbox coordinates in JSON format.'
    img_url = "./assets/spatial_understanding/lots_of_cars.png"
    
    print(f"Image: {img_url}")
    print(f"Prompt: {prompt}")
    
    if os.path.exists(img_url):
        model_response = inference_with_local_api(img_url, prompt)
        print("\nResponse:", model_response)
        
        response = Image.open(img_url)
        response.thumbnail([640,640], Image.Resampling.LANCZOS)
        plot_bounding_boxes(response, model_response)
    else:
        print(f"Image file {img_url} does not exist. Please download sample images to run this example.")
    
    print("="*70 + "\n")


# Example 4: Detecting vehicles with additional key information
def example_4_vehicle_attributes():
    """Example 4: Detecting vehicles with attributes"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Detecting Vehicles with Additional Information")
    print("="*70)

    prompt = 'locate every instance that belongs to the following categories: "vehicle". For each vehicle, report bbox coordinates, vehicle type and vehicle color in JSON format like this: {"bbox_2d": [x1, y1, x2, y2], "label": "vehicle", "type": "car, bus, truck, bicycle, ...", "color": "vehicle_color"}'
    img_url = "./assets/spatial_understanding/drone_cars2.png"
    
    print(f"Image: {img_url}")
    print(f"Prompt: {prompt}")
    
    if os.path.exists(img_url):
        model_response = inference_with_local_api(img_url, prompt)
        print("\nResponse:", model_response)
        
        response = Image.open(img_url)
        response.thumbnail([640,640], Image.Resampling.LANCZOS)
        plot_bounding_boxes(response, model_response)
    else:
        print(f"Image file {img_url} does not exist. Please download sample images to run this example.")
    
    print("="*70 + "\n")


# Example 5: Pointing out people in football field
def example_5_football_field_points():
    """Example 5: Pointing out people in football field"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Pointing out People in Football Field")
    print("="*70)

    prompt = '''Locate every person inside the football field with points, report their point coordinates, role(player, referee or unknown) and shirt color in JSON format like this: {"point_2d": [x, y], "label": "person", "role": "player/referee/unknown", "shirt_color": "the person's shirt color"}'''
    img_url = "./assets/spatial_understanding/football_field.jpg"
    
    print(f"Image: {img_url}")
    print(f"Prompt: {prompt}")
    
    if os.path.exists(img_url):
        model_response = inference_with_local_api(img_url, prompt)
        print("\nResponse:", model_response)
        
        response = Image.open(img_url)
        response.thumbnail([640,640], Image.Resampling.LANCZOS)
        plot_points_json(response, model_response)
    else:
        print(f"Image file {img_url} does not exist. Please download sample images to run this example.")
    
    print("="*70 + "\n")


# Main execution function
def main():
    """
    Main function to run examples.
    """
    print("="*60)
    print("Spatial Understanding with Local LLM")
    print("="*60)
    print(f"Model: {LOCAL_MODEL_ID}")
    print(f"Endpoint: {os.getenv('OPENAI_BASE_HTTP_API_URL')}")
    print("="*60)

    # Test API connection first
    if not test_api_connection():
        print("\nCannot proceed without API connection.")
        return

    print("\nTo run spatial understanding examples, make sure you have the required assets in ./assets/spatial_understanding/")
    
    # Run examples if assets exist
    print("\nRunning spatial understanding examples...")
    
    # Example 1: Multi-target detection
    example_1_multi_target_detection()
    
    # Example 2: Crowded scene
    example_2_crowded_scene()
    
    # Example 3: Drone view
    example_3_drone_view()
    
    # Example 4: Vehicle attributes
    example_4_vehicle_attributes()
    
    # Example 5: Football field points
    example_5_football_field_points()

    print("Done!")


if __name__ == "__main__":
    main()