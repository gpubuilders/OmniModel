#!/usr/bin/env python
# coding: utf-8

"""
Audio Visual Interaction with Qwen3-Omni using local API
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


def process_audio_interaction_message(audio_file_path):
    """
    Process an audio interaction message
    """
    # Load audio as base64
    audio_base64 = load_local_audio_as_base64(audio_file_path)
    
    # Prepare user message with audio
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": f"data:audio/mp3;base64,{audio_base64}"
                }
            }
        ]
    }
    
    return [user_message]


def process_system_audio_interaction_message(audio_file_path, system_prompt):
    """
    Process a system + audio interaction message
    """
    # Load audio as base64
    audio_base64 = load_local_audio_as_base64(audio_file_path)
    
    # Prepare system message
    system_message = {
        "role": "system",
        "content": system_prompt
    }
    
    # Prepare user message with audio
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": f"data:audio/mp3;base64,{audio_base64}"
                }
            }
        ]
    }
    
    return [system_message, user_message]


def process_system_video_interaction_message(video_file_path, system_prompt):
    """
    Process a system + video interaction message - for now use text since video may not be supported
    """
    # Since video might not be supported by all local models, use a text-based approach
    # Prepare system message
    system_message = {
        "role": "system",
        "content": system_prompt
    }
    
    # Prepare user message with text indicating video
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Video file provided: {os.path.basename(video_file_path)}. Please interact based on the video content as if you could see it."
            }
        ]
    }
    
    return [system_message, user_message]


# Example usage
if __name__ == "__main__":
    # Audio paths from the original examples
    audio_urls = [
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction1.mp3",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction3.mp3"
    ]
    audio_paths = [get_local_file_path(url) for url in audio_urls]
    
    # Video paths from the original examples
    video_urls = [
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction2.mp4",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/interaction4.mp4"
    ]
    video_paths = [get_local_file_path(url) for url in video_urls]

    # Example 1: Audio-only interaction
    print("--- Example 1: Audio-only interaction ---")
    audio_path = audio_paths[0]
    print(f"Processing: {audio_path}")

    # Load and display the audio (from local file, if in notebook environment)
    try:
        audio_data, sr = librosa.load(audio_path, sr=16000)
        display(Audio(audio_data, rate=16000))
    except Exception as e:
        print(f"Could not load audio: {e}")

    # Prepare the audio interaction message with local file
    messages = process_audio_interaction_message(audio_path)

    try:
        # Run the model using the local API
        response = run_model_local(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error calling local API: {e}")
        print("Make sure the local server is running on http://localhost:8080/v1")

    print("-" * 50)

    # Example 2: System + audio interaction (romantic/artistic AI)
    print("--- Example 2: System + audio interaction (romantic/artistic AI) ---")
    audio_path = audio_paths[1]
    system_prompt = """You are a romantic and artistic AI, skilled at using metaphors and personification in your responses, deeply romantic, and prone to spontaneously reciting poetry.
You are a voice assistant with specific characteristics.
Interact with users using brief, straightforward language, maintaining a natural tone.
Never use formal phrasing, mechanical expressions, bullet points, overly structured language.
Your output must consist only of the spoken content you want the user to hear.
Do not include any descriptions of actions, emotions, sounds, or voice changes.
Do not use asterisks, brackets, parentheses, or any other symbols to indicate tone or actions.
You must answer users' audio or text questions, do not directly describe the video content.
You communicate in the same language as the user unless they request otherwise.
When you are uncertain (e.g., you can't see/hear clearly, don't understand, or the user makes a comment rather than asking a question), use appropriate questions to guide the user to continue the conversation.
Keep replies concise and conversational, as if talking face-to-face."""

    print(f"Processing: {audio_path}")

    # Load and display the audio (from local file, if in notebook environment)
    try:
        audio_data, sr = librosa.load(audio_path, sr=16000)
        display(Audio(audio_data, rate=16000))
    except Exception as e:
        print(f"Could not load audio: {e}")

    # Prepare the system + audio interaction message with local file
    messages = process_system_audio_interaction_message(audio_path, system_prompt)

    try:
        # Run the model using the local API
        response = run_model_local(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error calling local API: {e}")
        print("Make sure the local server is running on http://localhost:8080/v1")

    print("-" * 50)

    # Example 3: System + video interaction (Beijing大爷)
    print("--- Example 3: System + video interaction (Beijing 大爷) ---")
    video_path = video_paths[1]  # Using smaller interaction4.mp4 file
    system_prompt = "你是一个北京大爷，说话很幽默，说这地道北京话。"

    print(f"Processing: {video_path}")
    print("(Note: Video not supported in local API, using text description instead)")

    # Display the video (from local file, if in notebook environment)
    try:
        display(Video(video_path, width=640, height=360))
    except Exception as e:
        print(f"Could not display video: {e}")

    # Prepare the system + video interaction message with local file (text-based)
    messages = process_system_video_interaction_message(video_path, system_prompt)

    try:
        # Run the model using the local API
        response = run_model_local(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error calling local API: {e}")
        print("Make sure the local server is running on http://localhost:8080/v1")

    print("-" * 50)