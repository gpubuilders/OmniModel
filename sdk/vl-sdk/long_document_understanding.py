#!/usr/bin/env python
# coding: utf-8

# ## Long Document Understanding with Qwen3-VL (Local Version)
#
# In this notebook, we delve into the capabilities of the Qwen3-VL model for understanding **long document with hundreds of pages**. Our objective is to showcase how this advanced model can be applied to long/full-PDF document analysis scenarios.
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
    "qwen-vl-utils",
    "pdf2image",
    "torch",
    "transformers",
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


import math
import hashlib
import requests

import numpy as np
from PIL import Image
from pdf2image import convert_from_path


def get_pdf_images(pdf_path, dpi=144, cache_dir='cache'):
    os.makedirs(cache_dir, exist_ok=True)

    # Create a hash for the PDF path to use in cache filenames
    pdf_hash = hashlib.md5(pdf_path.encode('utf-8')).hexdigest()

    # Only handle local file paths (no URLs)
    pdf_file_path = pdf_path

    # Check for cached images
    images_cache_file = os.path.join(cache_dir, f'{pdf_hash}_{dpi}_images.npy')
    if os.path.exists(images_cache_file):
        images = np.load(images_cache_file, allow_pickle=True)
        pil_images = [Image.fromarray(image) for image in images]
        print(f"Load {len(images)} pages from cache: {images_cache_file}.")
        return pdf_file_path, pil_images

    # Convert PDF to images if not cached
    print(f"Converting PDF to images at {dpi} DPI...")
    pil_images = convert_from_path(pdf_file_path, dpi=dpi)

    # image file size control
    resize_pil_images = []
    for img in pil_images:
        width, height = img.size
        max_side = max(width, height)
        max_side_value = 1500
        if max_side > max_side_value:
            img = img.resize((width * max_side_value // max_side, height * max_side_value // max_side))
        resize_pil_images.append(img)
    pil_images = resize_pil_images

    images = [np.array(img) for img in pil_images]

    # Save to cache
    np.save(images_cache_file, images)
    print(f"Converted and cached {len(images)} pages to {images_cache_file}.")

    return pdf_file_path, pil_images


def create_image_grid(pil_images, num_columns=8):

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

import base64
from io import BytesIO

def image_to_base64(img, format="PNG"):

    buffered = BytesIO()
    img.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return img_base64


# API Inference function using local OpenAI-compatible server

def inference_with_local_api(images, prompt, sys_prompt="", model_id=LOCAL_MODEL_ID, min_pixels=590*32*32, max_pixels=730*32*32):
    """
    Perform inference using local OpenAI-compatible API server.

    Args:
        images: List of PIL images to analyze
        prompt: Text prompt for the model
        sys_prompt: System prompt (optional)
        model_id: Model ID to use for inference
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing

    Returns:
        str: Generated text response from the model
    """
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_HTTP_API_URL'),
    )
    print("Send {} pages to the model... \nWaiting for response...".format(len(images)))

    content_list = []
    for image in images:
        base64_image = image_to_base64(image)
        content_list.append(
            {
                "type": "image_url",
                # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                # PNG image:  f"data:image/png;base64,{base64_image}"
                # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                # WEBP image: f"data:image/webp;base64,{base64_image}"
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            },
        )
    content_list.append({"type": "text", "text": prompt})
    messages = [
        # {
        #     "role": "system",
        #     "content": [{"type":"text","text": sys_prompt}]
        # },
        {
            "role": "user",
            "content": content_list
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


def example_1_long_document_summarization():
    """Example 1: Long document summarization"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Long Document Summarization")
    print("="*70)

    # Using a local PDF document path.
    local_pdf_path = "./assets/long_document_understanding/Qwen2.5-VL.pdf"
    prompt = "Please summarize the key contributions of this paper based on its abstract and introduction."

    print(f"PDF Path: {local_pdf_path}")
    print(f"Prompt: {prompt}")

    if os.path.exists(local_pdf_path):
        try:
            # This will load the local PDF, convert its pages to images, and then run inference.
            pdf_path, images = get_pdf_images(local_pdf_path, dpi=144)

            # You can use this to visualize documents in thumbnail format.
            image_grid = create_image_grid(images, num_columns=8)
            display(image_grid.resize((1000, 1000)))

            response = inference_with_local_api(images, prompt)
            print("\nResponse:")
            display(Markdown(response))
        except Exception as e:
            print(f"Error in example: {e}")
    else:
        print(f"Local PDF file {local_pdf_path} does not exist. Please download sample documents to run this example.")

    print("="*70 + "\n")


def example_2_count_tables():
    """Example 2: Counting tables in a document"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Counting Tables in a Document")
    print("="*70)

    # Using a local PDF document path.
    local_pdf_path = "./assets/long_document_understanding/fox_got_merge_code.pdf"
    prompt = "How many tables?"

    print(f"PDF Path: {local_pdf_path}")
    print(f"Prompt: {prompt}")

    if os.path.exists(local_pdf_path):
        try:
            # This will load the local PDF, convert its pages to images, and then run inference with API.
            pdf_path, images = get_pdf_images(local_pdf_path, dpi=144)

            # You can use this to visualize documents in thumbnail format.
            image_grid = create_image_grid(images, num_columns=8)
            display(image_grid.resize((1500, 1100)))
            print(f"First page size: {images[0].size}")

            response = inference_with_local_api(images, prompt)
            print("\nResponse:")
            display(Markdown(response))
        except Exception as e:
            print(f"Error in example: {e}")
    else:
        print(f"Local PDF file {local_pdf_path} does not exist. Please download sample documents to run this example.")

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
    print("Long Document Understanding with Local LLM")
    print("="*60)
    print(f"Model: {LOCAL_MODEL_ID}")
    print(f"Endpoint: {os.getenv('OPENAI_BASE_HTTP_API_URL')}")
    print("="*60)

    # Test API connection first
    if not test_api_connection():
        print("\nCannot proceed without API connection.")
        return

    print("\nRunning long document understanding examples...")
    
    # Example 1: Long document summarization
    example_1_long_document_summarization()
    
    # Example 2: Counting tables in a document
    example_2_count_tables()

    print("Done!")


if __name__ == "__main__":
    main()