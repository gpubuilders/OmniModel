#!/usr/bin/env python
# coding: utf-8

"""
Image Math with Qwen3-Omni using local API
This version uses requests to connect to a local API endpoint instead of transformers/vLLM
"""

import os
import requests
import json
import base64
from io import BytesIO
import numpy as np
from PIL import Image

# For displaying content in notebook environment
try:
    from IPython.display import Image as IPyImage, display
except ImportError:
    # Define dummy functions if IPython is not available
    def display(content):
        print(content)
    
    def IPyImage(data, width=None, height=None):
        return f"Image(width={width}, height={height})"


def get_local_file_path(url):
    """Convert URL to local file path in assets directory"""
    # Extract filename from URL
    filename = url.split('/')[-1]
    return f"assets/{filename}"


def load_local_image_as_base64(file_path):
    """Load a local image file and return it as base64 encoded string"""
    with open(file_path, 'rb') as f:
        image_bytes = f.read()
        return base64.b64encode(image_bytes).decode('utf-8')


def run_model_local(messages, model="Qwen3-Omni-10k"):
    """
    Run the model using local API endpoint
    """
    # API endpoint
    api_url = "http://localhost:8080/v1/chat/completions"
    
    # Prepare the payload
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 8192,
        "stream": False  # For simplicity, not streaming
    }
    
    # Make the request
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]


def process_image_math_message(image_file_path, math_problem, options):
    """
    Process an image math problem with options
    """
    # Load image as base64
    image_base64 = load_local_image_as_base64(image_file_path)
    
    # Prepare user message with math problem, image, and options
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": math_problem
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            },
            {
                "type": "text",
                "text": options
            }
        ]
    }
    
    return [user_message]


# Example usage
if __name__ == "__main__":
    # Image paths from the original examples
    image_urls = [
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/5195.jpg",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/4181.jpg"
    ]
    image_paths = [get_local_file_path(url) for url in image_urls]

    # Example 1: Lawn sprinkler problem
    print("--- Example 1: Lawn sprinkler problem ---")
    image_path = image_paths[0]
    math_problem = "The 3-arm lawn sprinkler of Fig. P3.114 receives 20°C water through the center at 2.7 m^3/hr. If collar friction is neglected, what is the steady rotation rate in rev/min for $\\theta $ = 40°?"
    options = "A, 317 rev/min, B, 414 rev/min, C, 400 rev/min, D, NaN, choose one of the options."

    print(f"Processing: {image_path}")

    # Display the image (from local file, if in notebook environment)
    try:
        display(IPyImage(image_path, width=640, height=360))
    except Exception as e:
        print(f"Could not display image: {e}")

    # Prepare the image math message with local file
    messages = process_image_math_message(image_path, math_problem, options)

    try:
        # Run the model using the local API
        response = run_model_local(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error calling local API: {e}")
        print("Make sure the local server is running on http://localhost:8080/v1")

    print("-" * 50)

    # Example 2: Training set cost function problem
    print("--- Example 2: Training set cost function problem ---")
    image_path = image_paths[1]
    math_problem = "Suppose we have a training set with m=3 examples,plotted below.Our hypothes is representation is $h_{\\theta}(x) = \\theta_1x$, with parameter $\\theta_1$. The cost function $J(\\theta_1)$ is $J(\\theta_1)$=${1\\over2m}\\sum\\nolimits_{i=1}^m(h_\\theta(x^i) - y^i)^2$. What is $J(0)$?"
    options = "Options:\nA. 0\nB. 1/6\nC. 1\nD. 14/6\nPlease select the correct answer from the options above."

    print(f"Processing: {image_path}")

    # Display the image (from local file, if in notebook environment)
    try:
        display(IPyImage(image_path, width=640, height=360))
    except Exception as e:
        print(f"Could not display image: {e}")

    # Prepare the image math message with local file
    messages = process_image_math_message(image_path, math_problem, options)

    try:
        # Run the model using the local API
        response = run_model_local(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error calling local API: {e}")
        print("Make sure the local server is running on http://localhost:8080/v1")

    print("-" * 50)