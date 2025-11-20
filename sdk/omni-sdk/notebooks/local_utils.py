#!/usr/bin/env python
# coding: utf-8

"""
Script to create local versions of all cookbook examples
"""

import os
import re
import requests
import json
import base64
from io import BytesIO
import numpy as np
from PIL import Image

# For displaying content in notebook environment
try:
    from IPython.display import display
except ImportError:
    def display(content):
        print(content)


def get_local_file_path(url):
    """Convert URL to local file path in assets directory"""
    # Extract filename from URL
    filename = url.split('/')[-1]
    return f"assets/{filename}"


def load_local_audio_as_base64(file_path):
    """Load a local audio file and return it as base64 encoded string"""
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()
        return base64.b64encode(audio_bytes).decode('utf-8')


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


def process_multimodal_message(content_file_path, prompt, content_type="image"):
    """
    Process a multimodal message with content (audio/image) and text
    """
    if content_type == "audio":
        content_base64 = load_local_audio_as_base64(content_file_path)
        content_url = f"data:audio/mp3;base64,{content_base64}"
        content_type_key = "audio_url"
    elif content_type == "image":
        content_base64 = load_local_image_as_base64(content_file_path)
        content_url = f"data:image/jpeg;base64,{content_base64}"
        content_type_key = "image_url"
    else:
        # For other types, default to image handling
        content_base64 = load_local_image_as_base64(content_file_path)
        content_url = f"data:image/jpeg;base64,{content_base64}"
        content_type_key = "image_url"
    
    # Prepare messages in the format expected by the API
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": content_type_key,
                    content_type_key: {
                        "url": content_url
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    return messages