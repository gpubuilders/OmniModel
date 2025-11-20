#!/usr/bin/env python
# coding: utf-8

"""
Music Analysis with Qwen3-Omni using local API
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
    from IPython.display import Audio, display
except ImportError:
    # Define dummy functions if IPython is not available
    def display(content):
        print(content)
    
    def Audio(data, rate=None):
        return f"Audio(rate={rate})"


def get_local_file_path(url):
    """Convert URL to local file path in assets directory"""
    # Extract filename from URL - handling the Chinese filename too
    filename = url.split('/')[-1]
    return f"assets/{filename}"


def load_local_audio_as_base64(file_path):
    """Load a local audio file and return it as base64 encoded string"""
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()
        return base64.b64encode(audio_bytes).decode('utf-8')


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


def process_music_analysis_message(audio_file_path, analysis_prompt):
    """
    Process a music analysis message
    """
    # Load audio as base64
    audio_base64 = load_local_audio_as_base64(audio_file_path)
    
    # Prepare user message with audio and analysis prompt
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": f"data:audio/mp3;base64,{audio_base64}"
                }
            },
            {
                "type": "text",
                "text": analysis_prompt
            }
        ]
    }
    
    return [user_message]


# Example usage
if __name__ == "__main__":
    # Audio paths from the original examples
    audio_urls = [
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/音乐风格-调性.mp3",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/37573.mp3",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/353349.mp3"
    ]
    audio_paths = [get_local_file_path(url) for url in audio_urls]

    # Example 1: Music style analysis (Chinese)
    print("--- Example 1: Music style analysis (Chinese) ---")
    audio_path = audio_paths[0]
    analysis_prompt = "请分析这是什么风格的音乐？"

    print(f"Processing: {audio_path}")

    # Load and display the audio (from local file, if in notebook environment)
    try:
        audio_data, sr = librosa.load(audio_path, sr=16000)
        display(Audio(audio_data, rate=16000))
    except Exception as e:
        print(f"Could not load audio: {e}")

    # Prepare the music analysis message with local file
    messages = process_music_analysis_message(audio_path, analysis_prompt)

    try:
        # Run the model using the local API
        response = run_model_local(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error calling local API: {e}")
        print("Make sure the local server is running on http://localhost:8080/v1")

    print("-" * 50)

    # Example 2: Detailed music analysis (English)
    print("--- Example 2: Detailed music analysis (English) ---")
    audio_path = audio_paths[1]
    analysis_prompt = "Describe the style, rhythm, dynamics, and expressed emotions of this piece of music. Identify the instruments used and suggest possible scenarios from which this music might originate."

    print(f"Processing: {audio_path}")

    # Load and display the audio (from local file, if in notebook environment)
    try:
        audio_data, sr = librosa.load(audio_path, sr=16000)
        display(Audio(audio_data, rate=16000))
    except Exception as e:
        print(f"Could not load audio: {e}")

    # Prepare the music analysis message with local file
    messages = process_music_analysis_message(audio_path, analysis_prompt)

    try:
        # Run the model using the local API
        response = run_model_local(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error calling local API: {e}")
        print("Make sure the local server is running on http://localhost:8080/v1")

    print("-" * 50)

    # Example 3: Appreciative music analysis
    print("--- Example 3: Appreciative music analysis ---")
    audio_path = audio_paths[2]
    analysis_prompt = "Write an appreciative description for this piece of music. Identifying its style and genre. Analyze the collaborative patterns of different instruments in audio and explain their impact on the overall atmosphere."

    print(f"Processing: {audio_path}")

    # Load and display the audio (from local file, if in notebook environment)
    try:
        audio_data, sr = librosa.load(audio_path, sr=16000)
        display(Audio(audio_data, rate=16000))
    except Exception as e:
        print(f"Could not load audio: {e}")

    # Prepare the music analysis message with local file
    messages = process_music_analysis_message(audio_path, analysis_prompt)

    try:
        # Run the model using the local API
        response = run_model_local(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error calling local API: {e}")
        print("Make sure the local server is running on http://localhost:8080/v1")

    print("-" * 50)