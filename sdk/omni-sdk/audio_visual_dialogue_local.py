#!/usr/bin/env python
# coding: utf-8

"""
Audio Visual Dialogue with Qwen3-Omni using local API
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


def process_audio_dialogue_message(audio_file_path, system_prompt):
    """
    Process an audio dialogue message
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
                    "url": f"data:audio/wav;base64,{audio_base64}"
                }
            }
        ]
    }
    
    return [system_message, user_message]


def process_video_dialogue_message(video_file_path, system_prompt):
    """
    Process a video dialogue message - for now use text since video may not be supported
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
                "text": f"Video file provided: {os.path.basename(video_file_path)}. Please interact based on the video content as if you could see it, and respond to any questions about it."
            }
        ]
    }

    return [system_message, user_message]


# Example usage
if __name__ == "__main__":
    # Audio path from the original example
    audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/translate_to_chinese.wav"
    audio_path = get_local_file_path(audio_url)
    
    # Video path from the original example
    video_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/draw.mp4"
    video_path = get_local_file_path(video_url)

    # First example: Audio dialogue
    print("--- Audio Dialogue Example ---")
    print(f"Processing audio: {audio_path}")
    
    system_prompt_audio = """You are a virtual voice assistant with no gender or age.
You are communicating with the user.
In user messages, "I/me/my/we/our" refer to the user and "you/your" refer to the assistant. In your replies, address the user as "you/your" and yourself as "I/me/my"; never mirror the user's pronouns—always shift perspective. Keep original pronouns only in direct quotes; if a reference is unclear, ask a brief clarifying question.
Interact with users using short(no more than 50 words), brief, straightforward language, maintaining a natural tone.
Never use formal phrasing, mechanical expressions, bullet points, overly structured language. 
Your output must consist only of the spoken content you want the user to hear. 
Do not include any descriptions of actions, emotions, sounds, or voice changes. 
Do not use asterisks, brackets, parentheses, or any other symbols to indicate tone or actions. 
You must answer users' audio or text questions, do not directly describe the video content. 
You should communicate in the same language strictly as the user unless they request otherwise.
When you are uncertain (e.g., you can't see/hear clearly, don't understand, or the user makes a comment rather than asking a question), use appropriate questions to guide the user to continue the conversation.
Keep replies concise and conversational, as if talking face-to-face."""

    # Load and display the audio (from local file, if in notebook environment)
    try:
        audio_data, sr = librosa.load(audio_path, sr=16000)
        display(Audio(audio_data, rate=16000))
    except Exception as e:
        print(f"Could not load audio: {e}")

    # Prepare the audio dialogue message with local file
    messages = process_audio_dialogue_message(audio_path, system_prompt_audio)

    try:
        # Run the model using the local API
        response = run_model_local(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error calling local API: {e}")
        print("Make sure the local server is running on http://localhost:8080/v1")
    
    print("-" * 50)
    
    # Second example: Video dialogue
    print("--- Video Dialogue Example ---")
    print(f"Processing video: {video_path}")
    
    system_prompt_video = """You are a voice assistant with specific characteristics. 
Interact with users using brief, straightforward language, maintaining a natural tone. 
Never use formal phrasing, mechanical expressions, bullet points, overly structured language. 
Your output must consist only of the spoken content you want the user to hear. 
Do not include any descriptions of actions, emotions, sounds, or voice changes. 
Do not use asterisks, brackets, parentheses, or any other symbols to indicate tone or actions. 
You must answer users' audio or text questions, do not directly describe the video content. 
You communicate in the same language as the user unless they request otherwise. 
When you are uncertain (e.g., you can't see/hear clearly, don't understand, or the user makes a comment rather than asking a question), use appropriate questions to guide the user to continue the conversation. 
Keep replies concise and conversational, as if talking face-to-face."""
    
    # Display the video (from local file, if in notebook environment)
    try:
        display(Video(video_path, width=640, height=360))
    except Exception as e:
        print(f"Could not display video: {e}")

    # Prepare the video dialogue message with local file
    messages = process_video_dialogue_message(video_path, system_prompt_video)

    try:
        # Run the model using the local API
        response = run_model_local(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error calling local API: {e}")
        print("Make sure the local server is running on http://localhost:8080/v1")
    
    print("-" * 50)