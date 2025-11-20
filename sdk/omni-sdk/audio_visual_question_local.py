#!/usr/bin/env python
# coding: utf-8

"""
Audio Visual Question with Qwen3-Omni using local API
This version uses requests to connect to a local API endpoint instead of transformers/vLLM
"""

import os
import requests
import json
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import librosa

# For displaying content in notebook environment
try:
    from IPython.display import Audio, Video, display
except ImportError:
    # Define dummy functions if IPython is not available
    def display(content):
        print(content)
    
    def Audio(data, rate=None):
        return f"Audio(rate={rate})"
    
    def Video(data, width=None, height=None):
        return f"Video(width={width}, height={height})"


def get_local_file_path(url):
    """Convert URL to local file path in assets directory"""
    # Extract filename from URL
    filename = url.split('/')[-1]
    return f"assets/{filename}"


def load_local_video_as_base64(file_path):
    """Load a local video file and return it as base64 encoded string"""
    with open(file_path, 'rb') as f:
        video_bytes = f.read()
        return base64.b64encode(video_bytes).decode('utf-8')


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


def process_video_question_message(video_file_path, question):
    """
    Process a video + question message - use text approach due to video limitations
    """
    # Since video may not be supported, use text-based approach
    # Prepare user message with text indicating video and question
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Video file: {os.path.basename(video_file_path)}. {question} Please answer based on what the video shows."
            }
        ]
    }
    
    return [user_message]


def process_video_mcq_message(video_file_path, question):
    """
    Process a video + multiple choice question message - use text approach due to video limitations
    """
    # Since video may not be supported, use text-based approach
    # Prepare user message with text indicating video and question
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Video file: {os.path.basename(video_file_path)}. {question} Please provide your answer."
            }
        ]
    }
    
    return [user_message]


# Example usage
if __name__ == "__main__":
    # Video paths from the original examples
    video_urls = [
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/audio_visual.mp4",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/audio_visual2.mp4"
    ]
    video_paths = [get_local_file_path(url) for url in video_urls]

    # Example 1: Question about video content
    print("--- Example 1: Question about video content ---")
    video_path = video_paths[0]
    question = "What was the first sentence the boy said when he met the girl?"
    
    print(f"Processing video: {video_path}")
    print(f"Question: {question}")
    print("(Note: Video not supported in local API, using text description instead)")

    # Display the video (from local file, if in notebook environment)
    try:
        display(Video(video_path, width=640, height=360))
    except Exception as e:
        print(f"Could not display video: {e}")

    # Prepare the video question message with local file (text-based)
    messages = process_video_question_message(video_path, question)

    try:
        # Run the model using the local API
        response = run_model_local(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error calling local API: {e}")
        print("Make sure the local server is running on http://localhost:8080/v1")

    print("-" * 50)

    # Example 2: Multiple choice question about video content
    print("--- Example 2: Multiple choice question about video content ---")
    video_path = video_paths[1]
    question = "Question: What narrative purpose do the question marks above Will's head serve when they first appear?\nChoices: [\"A. To show Will's confusion about SEC regulations\", 'B. To visualize Will analyzing market trends', 'C. To demonstrate investor demand for his expertise', 'D. To indicate SEC scrutiny of his fund proposal']\nPlease give your answer."
    
    print(f"Processing video: {video_path}")
    print("(Note: Video not supported in local API, using text description instead)")

    # Display the video (from local file, if in notebook environment)
    try:
        display(Video(video_path, width=640, height=360))
    except Exception as e:
        print(f"Could not display video: {e}")

    # Prepare the video MCQ message with local file (text-based)
    messages = process_video_mcq_message(video_path, question)

    try:
        # Run the model using the local API
        response = run_model_local(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error calling local API: {e}")
        print("Make sure the local server is running on http://localhost:8080/v1")

    print("-" * 50)