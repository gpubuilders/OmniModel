#!/usr/bin/env python
# coding: utf-8

# ### Video Understanding with Qwen3-VL (Local Version)
# 
# In this notebook, we delve into the capabilities of the **Qwen3-VL** model for video understanding tasks. 
# This version has been modified to use a local LLM server instead of DashScope API.

# #### [Setup]
# 
# Configure local LLM server connection

import os
import sys
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


# Load video frames and timestamps

import math
import hashlib
import requests
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu


def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")


def get_video_frames(video_path, num_frames=128, cache_dir='.cache'):
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

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)
    
    return video_file_path, frames, timestamps


def create_image_grid(images, num_columns=8):
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


# API Inference function using local OpenAI-compatible server

def inference_with_local_api(
    video,
    prompt,
    model_id=LOCAL_MODEL_ID,
    video_type='url'
):
    """
    Perform inference using local OpenAI-compatible API server.
    
    Args:
        video: Video input (URL string or dict with frame_list)
        prompt: Text prompt for the model
        model_id: Model ID from your local server
        video_type: Either 'url' or 'frame_list'
    
    Returns:
        str: Generated text response from the model
    """
    if video_type == 'url':
        video_msg = {"type": "video_url", "video_url": {"url": video}}
    elif video_type == 'frame_list':
        video_msg = {"type": "video", "video": video['frame_list']}
    
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_HTTP_API_URL'),
    )    
    
    messages = [
        {
            "role": "user",
            "content": [
                video_msg,
                {"type": "text", "text": prompt},
            ]
        }
    ]
    
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
        )
        print(completion)
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error during API call: {e}")
        raise


# #### [Usage]
# 
# Once the local API server is running, you can provide video inputs in **two formats**:
# 
# 1. **Video URL (`video_url`)** — A file path or publicly accessible HTTP(S) URL pointing to a video file (e.g., MP4, AVI).  
#    ✅ Best for quick prototyping or when you want the model/API to handle video decoding internally.
# 2. **Frame List (`frame_list`)** — A list of PIL Image objects or file paths representing sampled frames from a video.  
#    ✅ Best for fine-grained control, preprocessing, or when you've already decoded the video.


# ### 1. Using Video URL - API Inference

def example_1_video_url():
    """Example 1: Using Video URL with local API"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Video URL Analysis (Chinese)")
    print("="*70)
    
    video_url = "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4"
    prompt = "请用表格总结一下视频中的商品特点 but in english"

    print(f"\nVideo URL: {video_url}")
    print(f"Prompt: {prompt}")
    print("\nExtracting frames...")
    
    # Optional: Display video frames
    video_path, frames, timestamps = get_video_frames(video_url, num_frames=64)
    image_grid = create_image_grid(frames, num_columns=8)
    
    if IN_JUPYTER:
        display(image_grid.resize((640, 640)))
    else:
        print(f"✓ Extracted {len(frames)} frames")
        print(f"  Grid size: {image_grid.size}")
        print(f"  You can save this grid with: image_grid.save('output_grid.jpg')")

    print("\nSending request to local API...")
    # Call local API
    response = inference_with_local_api(video_url, prompt, video_type='url')
    
    print("\n" + "-"*70)
    print("RESPONSE:")
    print("-"*70)
    print(response)
    print("="*70 + "\n")
    
    return response


# ### 2. Using Video URL - Another Example

def example_2_video_url_timestamps():
    """Example 2: Video analysis with timestamp localization"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Video Event Localization with Timestamps")
    print("="*70)
    
    video_url = "https://ofasys-multimodal-wlcb-3.oss-cn-wulanchabu.aliyuncs.com/sibo.ssb/datasets/cookbook/ead2e3f0e7f836c9ec51236befdaf2d843ac13a6.mp4"
    prompt = "Localize a series of activity events in the video, output the start and end timestamp for each event, and describe each event with sentences. Provide the result in json format with 'mm:ss.ff' format for time depiction."

    print(f"\nVideo URL: {video_url}")
    print(f"Prompt: {prompt}")
    print("\nSending request to local API...")
    
    response = inference_with_local_api(video_url, prompt, video_type='url')
    
    print("\n" + "-"*70)
    print("RESPONSE:")
    print("-"*70)
    print(response)
    print("="*70 + "\n")
    
    return response


# ### 3. Using Frame List — API Inference

def example_3_frame_list():
    """Example 3: Using pre-extracted frame list"""
    # Base URL for pre-extracted video frames (public OSS bucket)
    video_frame_dir = 'https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/Qwen3VL/demo/video/demo_cooking'

    # Configure sampling: e.g., 0.25 FPS = 1 frame per 4 seconds
    sample_fps = 0.25  # or =1 

    video_frame_list = [f"{video_frame_dir}/{i}.000.jpg" for i in range(0, 1228, int(1/sample_fps))]
    
    video = {
        'frame_list': video_frame_list,
        'fps': str(sample_fps)
    }
    
    prompt = "Briefly describe the video."
    response = inference_with_local_api(video, prompt, video_type='frame_list')
    display(Markdown(response))


# ### 4. Using Frame List — Another Example

def example_4_frame_list_math():
    """Example 4: Frame list analysis"""
    video_frame_list = [
        f"https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/Qwen3VL/demo/video/validation_Math_6/{i}.000.jpg" 
        for i in range(0, 302, 2)
    ]

    video = {
        'frame_list': video_frame_list,
        'fps': '0.5'
    }
    
    prompt = "Describe this video."
    response = inference_with_local_api(video, prompt, video_type='frame_list')
    display(Markdown(response))


# ### 5. Using Interleaved Timestamp-Image Pairs — API Inference

def example_5_interleaved_timestamps():
    """Example 5: Spatial-temporal grounding with interleaved timestamp-image pairs"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<0.0 seconds>"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/Qwen3VL/demo/video/VidSTG_video0908val_fps1/2588643984_frames/2588643984_frame_00000000.jpg"
                    }
                },
                {
                    "type": "text",
                    "text": "<1.0 seconds>"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/Qwen3VL/demo/video/VidSTG_video0908val_fps1/2588643984_frames/2588643984_frame_00000030.jpg"
                    }
                },
                {
                    "type": "text",
                    "text": "<2.0 seconds>"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/Qwen3VL/demo/video/VidSTG_video0908val_fps1/2588643984_frames/2588643984_frame_00000060.jpg"
                    }
                },
                {
                    "type": "text",
                    "text": "<3.0 seconds>"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/Qwen3VL/demo/video/VidSTG_video0908val_fps1/2588643984_frames/2588643984_frame_00000090.jpg"
                    }
                },
                {
                    "type": "text",
                    "text": "<4.0 seconds>"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/Qwen3VL/demo/video/VidSTG_video0908val_fps1/2588643984_frames/2588643984_frame_00000120.jpg"
                    }
                },
                {
                    "type": "text",
                    "text": "<5.0 seconds>"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/Qwen3VL/demo/video/VidSTG_video0908val_fps1/2588643984_frames/2588643984_frame_00000150.jpg"
                    }
                },
                {
                    "type": "text",
                    "text": "<6.0 seconds>"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/Qwen3VL/demo/video/VidSTG_video0908val_fps1/2588643984_frames/2588643984_frame_00000180.jpg"
                    }
                },
                {
                    "type": "text",
                    "text": "<7.0 seconds>"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/Qwen3VL/demo/video/VidSTG_video0908val_fps1/2588643984_frames/2588643984_frame_00000210.jpg"
                    }
                },
                {
                    "type": "text",
                    "text": "Given the query \"there is a moving bicycle towards an adult in black in a path.\", for each frame, detect and localize the visual content described by the given textual query in JSON format. If the visual content does not exist in a frame, skip that frame. Output Format: [{\"time\": 1.0, \"bbox_2d\": [x_min, y_min, x_max, y_max], \"label\": \"\"}, {\"time\": 2.0, \"bbox_2d\": [x_min, y_min, x_max, y_max], \"label\": \"\"}, ...]."
                }
            ]
        }
    ]
    
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_HTTP_API_URL'),
    )
    
    completion = client.chat.completions.create(
        model=LOCAL_MODEL_ID,
        messages=messages,
        seed=125,
    )
    
    response = completion.choices[0].message.content
    return response, messages


# Helper functions for visualization

import json
import markdown
from bs4 import BeautifulSoup
from datetime import datetime
from PIL import ImageDraw
from io import BytesIO


def draw_bbox(image, bbox):
    """Draw bounding box on image"""
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline='red', width=4)
    return image


def create_image_grid_pil(pil_images, num_columns=8):
    """Create image grid from PIL images"""
    num_rows = math.ceil(len(pil_images) / num_columns)

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


def parse_json(response):
    """Parse JSON from markdown formatted response"""
    html = markdown.markdown(response, extensions=['fenced_code'])
    soup = BeautifulSoup(html, 'html.parser')
    json_text = soup.find('code').text

    data = json.loads(json_text)
    return data


def visualize_bbox_results(response, messages):
    """Visualize bounding box detection results"""
    results = parse_json(response)

    vis_images = []
    for content_idx, content in enumerate(messages[0]['content']):
        matched_result = None
        if content['type'] == 'text' and "seconds>" in content['text']:
            for result in results:
                time_str = str(result['time'])
                if time_str in content['text']:
                    matched_result = result

            image_url = messages[0]['content'][content_idx + 1]['image_url']['url']
            image = Image.open(BytesIO(requests.get(image_url).content))

            if matched_result is not None:
                image_width, image_height = image.size
                x_min, y_min, x_max, y_max = matched_result['bbox_2d']
                x_min = x_min / 1000 * image_width
                y_min = y_min / 1000 * image_height
                x_max = x_max / 1000 * image_width
                y_max = y_max / 1000 * image_height
                image = draw_bbox(image, [x_min, y_min, x_max, y_max])
                vis_images.append(image)
            else:
                vis_images.append(image)

    image_grid = create_image_grid_pil(vis_images, num_columns=2)
    display(image_grid.resize((640, 960)))


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
    Uncomment the example you want to run.
    """
    print("="*60)
    print("Video Understanding with Local LLM")
    print("="*60)
    print(f"Model: {LOCAL_MODEL_ID}")
    print(f"Endpoint: {os.getenv('OPENAI_BASE_HTTP_API_URL')}")
    print("="*60)
    
    # Test API connection first
    if not test_api_connection():
        return
    
    print("\nTo run video examples, uncomment one of the following:")
    print("  - example_1_video_url()           # Chinese video with table summary")
    print("  - example_2_video_url_timestamps() # Event localization with timestamps")
    print("  - example_3_frame_list()          # Cooking video frame list")
    print("  - example_4_frame_list_math()     # Math video frame list")
    print("  - example_5 with visualization    # Spatial-temporal grounding")
    print()
    
    # Uncomment the example you want to run:
    
    example_1_video_url()
    example_2_video_url_timestamps()
    example_3_frame_list()
    example_4_frame_list_math()
    
    # For example 5 with visualization:
    response, messages = example_5_interleaved_timestamps()
    # visualize_bbox_results(response, messages)
    
    print("Done!")


if __name__ == "__main__":
    main()