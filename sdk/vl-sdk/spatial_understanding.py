#!/usr/bin/env python
# coding: utf-8

"""
Spatial Understanding with Local Qwen3-VL API

This script demonstrates Qwen3-VL's ability to do more than just see objects. 
It understands their spatial layout, perceives what actions are possible ('affordances'), 
and uses this knowledge to reason like an embodied agent, using a local LLM service.
"""

import os
import subprocess
import sys
import json
import random
import io
import ast
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import xml.etree.ElementTree as ET
import base64
import requests
import hashlib
import math
import time
import re

# Import OpenAI here so it's available throughout the module
from openai import OpenAI

# Jupyter detection to handle display functions properly
def check_if_in_jupyter():
    """Check if running in Jupyter environment"""
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

IN_JUPYTER = check_if_in_jupyter()

def install_dependencies():
    """Install required packages if not already installed"""
    print("="*70)
    print("Checking and installing dependencies...")
    print("="*70)
    
    packages = [
        ("Pillow", "Pillow"),
        ("numpy", "numpy"),
        ("openai", "openai"),
        ("decord", "decord"),
    ]
    
    for import_name, package_name in packages:
        try:
            __import__(import_name)
            print(f"✓ {import_name} is already installed")
        except ImportError:
            print(f"Installing {import_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✓ {import_name} installed successfully")
    
    print("✓ All dependencies installed!")
    print()

# Install dependencies when the script runs
install_dependencies()

# Configuration for local API
API_KEY = "dummy-key"  # Using dummy key for local API
BASE_URL = "http://localhost:8080/v1"  # Local instance
MODEL_ID = "Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm"  # Local model

print("="*70)
print("Spatial Understanding with Local LLM")
print("="*70)
print(f"Model: {MODEL_ID}")
print(f"Endpoint: {BASE_URL}")
print("="*70)

def test_api_connection():
    """Test the local API connection"""
    print("\n" + "="*70)
    print("Testing API Connection...")
    print("="*70)

    try:
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        
        # Send a simple test message
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Hello, please respond with 'API connection successful!' and nothing else."}
                    ]
                }
            ],
            max_tokens=20
        )
        
        result = response.choices[0].message.content
        print(f"\n✓ API Connection Successful!")
        print(f"Model: {MODEL_ID}")
        print(f"Response: {result}")
        return True
        
    except Exception as e:
        print(f"\n❌ API Connection Failed: {str(e)}")
        return False

# @title Plotting Util
additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def decode_json_points(text: str):
    """Parse coordinate points from text format"""
    try:
        # Clean markdown markers
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]

        # Parse JSON
        data = json.loads(text)
        points = []
        labels = []

        for item in data:
            if "point_2d" in item:
                x, y = item["point_2d"]
                points.append([x, y])

                # Get label, use default if not present
                label = item.get("label", f"point_{len(points)}")
                labels.append(label)

        return points, labels

    except Exception as e:
        print(f"Error: {e}")
        return [], []


def plot_points(im, text):
    """Plot points on image based on coordinates from the model response"""
    img = im.copy()  # Make a copy to avoid modifying the original
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
        if not IN_JUPYTER:
            print("[Image would be displayed here in Jupyter]")
        else:
            img.show()
        return

    # Try to use a font - prefer system fonts
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=14)
    except:
        try:
            font = ImageFont.load_default()
        except:
            # Create a simple font if nothing else works
            font = ImageFont.load_default()

    for i, point in enumerate(points):
        color = colors[i % len(colors)]
        abs_x1 = int(point[0])/1000 * width
        abs_y1 = int(point[1])/1000 * height
        radius = 2
        draw.ellipse([(abs_x1 - radius, abs_y1 - radius), (abs_x1 + radius, abs_y1 + radius)], fill=color)
        draw.text((abs_x1 - 20, abs_y1 + 6), descriptions[i], fill=color, font=font)

    if not IN_JUPYTER:
        print("[Image would be displayed here in Jupyter]")
    else:
        img.show()

def encode_image(image_path):
    """Encode image to base64 format"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def spatial_inference_with_local_api(image_path, prompt, model_id=MODEL_ID):
    """Local API inference function"""
    base64_image = encode_image(image_path)
    
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # Determine image format
    image_format = image_path.split(".")[-1].lower()
    if image_format == 'jpg':
        image_format = 'jpeg'
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=1024
    )
    
    return response.choices[0].message.content

def example_1_spatial_relationships():
    """Example 1: Understand the Spatial Relationship Between Objects"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Spatial Relationships Between Objects")
    print("="*70)

    # Create assets directory if it doesn't exist
    os.makedirs("assets/spatial_understanding", exist_ok=True)
    
    # For now we'll use a placeholder - in a real scenario you would download actual images
    # Since we're working with local assets, let's assume images are already available
    image_path = "assets/spatial_understanding/spatio_case1.jpg"
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        print("   This example requires local spatial understanding assets.")
        print("   Please ensure the image exists in the assets directory.")
        print("="*70 + "\n")
        return
    
    prompt = "Which object, in relation to your current position, holds the farthest placement in the image?\nAnswer options:\nA.chair\nB.plant\nC.window\nD.tv stand."
    response = spatial_inference_with_local_api(image_path, prompt)

    print("Prompt:\n"+prompt)
    print("\nAnswer:\n"+response)
    
    # Display image info (in Jupyter it would show the image)
    if not IN_JUPYTER:
        print(f"Input image path: {image_path}")
        if os.path.exists(image_path):
            img = Image.open(image_path)
            print(f"Image size: {img.size}")
            print("[Image would be displayed here in Jupyter]")
    else:
        img = Image.open(image_path)
        img.show()

    print("="*70 + "\n")

def example_2_object_affordances():
    """Example 2: Perceive Object Affordances"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Object Affordances")
    print("="*70)

    os.makedirs("assets/spatial_understanding", exist_ok=True)
    
    # Example with spatial pointing
    image_path = "assets/spatial_understanding/spatio_case2_aff.png"
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        print("   This example requires local spatial understanding assets.")
        print("   Please ensure the image exists in the assets directory.")
        print("="*70 + "\n")
        return
    
    prompt = "Locate the free space on the white table on the right in this image. Output the point coordinates in JSON format."
    response = spatial_inference_with_local_api(image_path, prompt)

    print("Prompt:\n"+prompt)
    print("\nAnswer:\n"+response)
    
    if os.path.exists(image_path):
        img = Image.open(image_path)
        plot_points(img, response)
    
    # Second affordance example
    image_path2 = "assets/spatial_understanding/spatio_case2_aff2.png"
    
    if os.path.exists(image_path2):
        prompt2 = "Can the speaker fit behind the guitar?"
        response2 = spatial_inference_with_local_api(image_path2, prompt2)

        print("\nSecond Example:")
        print("Prompt:\n"+prompt2)
        print("\nAnswer:\n"+response2)
        
        img2 = Image.open(image_path2)
        img2_resized = img2.resize((img2.width//4, img2.height//4))
        
        if not IN_JUPYTER:
            print("[Image would be displayed here in Jupyter]")
        else:
            img2_resized.show()

    print("="*70 + "\n")

def example_3_spatial_reasoning_action_planning():
    """Example 3: Integrate Spatial Reasoning and Action Planning"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Spatial Reasoning and Action Planning")
    print("="*70)

    os.makedirs("assets/spatial_understanding", exist_ok=True)
    
    # First action planning example
    image_path = "assets/spatial_understanding/spatio_case2_plan.png"
    
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        print("   This example requires local spatial understanding assets.")
        print("   Please ensure the image exists in the assets directory.")
        print("="*70 + "\n")
        return
    
    prompt = "What color arrow should the robot follow to move the apple in between the green can and the orange? Choices: A. Red. B. Blue. C. Green. D. Orange."
    response = spatial_inference_with_local_api(image_path, prompt)

    print("Prompt:\n"+prompt)
    print("\nAnswer:\n"+response)
    
    # Display image info (in Jupyter it would show the image)
    if not IN_JUPYTER:
        print(f"Input image path: {image_path}")
        if os.path.exists(image_path):
            img = Image.open(image_path)
            print(f"Image size: {img.size}")
            print("[Image would be displayed here in Jupyter]")
    else:
        img = Image.open(image_path)
        img.show()

    # Second action planning example
    image_path2 = "assets/spatial_understanding/spatio_case2_plan2.png"
    
    if os.path.exists(image_path2):
        prompt2 = "Which motion can help change the coffee pod? Choices: A. A. B. B. C. C. D. D."
        response2 = spatial_inference_with_local_api(image_path2, prompt2)

        print("\nSecond Example:")
        print("Prompt:\n"+prompt2)
        print("\nAnswer:\n"+response2)
        
        if not IN_JUPYTER:
            print(f"Input image path: {image_path2}")
            if os.path.exists(image_path2):
                img2 = Image.open(image_path2)
                print(f"Image size: {img2.size}")
                print("[Image would be displayed here in Jupyter]")
        else:
            img2 = Image.open(image_path2)
            img2.show()

    print("="*70 + "\n")

# Video processing functions (for completeness)
def download_video(url, dest_path):
    """Download video from URL to local path"""
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")


def get_video_frames(video_path, num_frames=128, cache_dir='./assets/spatial_understanding/'):
    """Extract frames from video using decord"""
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    if video_path.startswith('http://') or video_path.startswith('https://'):
        video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
        if not os.path.exists(video_file_path):
            download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_file_path, ctx=cpu(0))
        total_frames = len(vr)

        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

        np.save(frames_cache_file, frames)
        np.save(timestamps_cache_file, timestamps)

        return video_file_path, frames, timestamps
    except Exception as e:
        print(f"Error processing video: {e}")
        return None, None, None


def create_image_grid(images, num_columns=8):
    """Create a grid of images from a list of image arrays"""
    if not images:
        return None
        
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = math.ceil(len(images) / num_columns)

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image


def video_spatial_inference_with_local_api(video_frames, prompt, model_id=MODEL_ID):
    """Local API inference function for video frames"""
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # Encode frames to base64
    frame_list = []
    for frame in video_frames:
        # Convert numpy array to PIL Image, then to base64
        pil_img = Image.fromarray(frame)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        base64_frame = base64.b64encode(buffer.getvalue()).decode('utf-8')
        frame_list.append(f"data:image/jpeg;base64,{base64_frame}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frame_list,
                    "fps": 2
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=1024
    )
    
    return response.choices[0].message.content


def example_4_video_spatial_reasoning():
    """Example 4: Video Spatial Reasoning (if video assets are available)"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Video Spatial Reasoning")
    print("="*70)
    
    # This is a placeholder since we don't have the video assets
    print("This example requires video assets for spatial reasoning in videos.")
    print("For a complete implementation, you would need to:")
    print("1. Download the video file")
    print("2. Extract frames using get_video_frames()")
    print("3. Process the frames with the model")
    print("4. Display the results as an image grid")
    
    # In a real implementation, you would use something like:
    # video_url = "https://xxxxx/42446167.mp4"
    # video_path, frames, timestamps = get_video_frames(video_url, num_frames=64)
    # if frames is not None:
    #     prompt = "These are frames of a video...\n[full prompt here]"
    #     response = video_spatial_inference_with_local_api(frames, prompt)
    #     image_grid = create_image_grid(frames, num_columns=8)
    #     print("Response:", response)
    #     print("[Video frames grid would be displayed in Jupyter]")
    
    print("="*70 + "\n")

# Simple test function to verify API connection
def simple_test():
    """Simple test to verify the system works"""
    print("\n" + "="*70)
    print("Testing Local API Connection...")
    print("="*70)

    try:
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello, please respond with 'Local API test successful!' and nothing else."}
                    ]
                }
            ],
            max_tokens=20
        )

        result = response.choices[0].message.content
        print(f"✓ Local API test successful!")
        print(f"Response: {result}")
        return True

    except Exception as e:
        print(f"❌ Local API test failed: {str(e)}")
        return False

def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Running spatial understanding examples...")
    print("="*70)
    
    # Test API connection
    api_connected = test_api_connection()
    if not api_connected:
        print("❌ Cannot proceed without API connection.")
        return
    
    # Run examples
    example_1_spatial_relationships()
    example_2_object_affordances()
    example_3_spatial_reasoning_action_planning()
    example_4_video_spatial_reasoning()
    
    print("Done!")

if __name__ == "__main__":
    main()