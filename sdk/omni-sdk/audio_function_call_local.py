#!/usr/bin/env python
# coding: utf-8

"""
Audio Function Call with Qwen3-Omni using local API
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


def process_function_call_message(audio_file_path):
    """
    Process a function call message with audio
    """
    # Load audio as base64
    audio_base64 = load_local_audio_as_base64(audio_file_path)
    
    # Prepare system message with tool definitions
    system_message = {
        "role": "system",
        "content": """You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{'type': 'function', 'function': {'name': 'web_search', 'description': 'Utilize the web search engine to retrieve relevant information based on multiple queries.', 'parameters': {'type': 'object', 'properties': {'queries': {'type': 'array', 'items': {'type': 'string', 'description': 'The search query.'}, 'description': 'The list of search queries.'}}, 'required': ['queries']}}}
{'type': 'function', 'function': {'name': 'car_ac_control', 'description': "Control the vehicle's air conditioning system to turn it on/off and set the target temperature", 'parameters': {'type': 'object', 'properties': {'temperature': {'type': 'number', 'description': 'Target set temperature in Celsius degrees'}, 'ac_on': {'type': 'boolean', 'description': 'Air conditioning status (true=on, false=off)'}}, 'required': ['temperature', 'ac_on']}}}
</tools>

For each function call, return a json object with function name and arguments within <invoke></invoke> XML tags:
<invoke>
{"name": <function-name>, "arguments": <args-json-object>}
</invoke>"""
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


# Example usage
if __name__ == "__main__":
    # Audio path from the original example
    audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/functioncall_case.wav"
    audio_path = get_local_file_path(audio_url)

    print(f"Processing: {audio_path}")

    # Load and display the audio (from local file, if in notebook environment)
    try:
        audio_data, sr = librosa.load(audio_path, sr=16000)
        display(Audio(audio_data, rate=16000))
    except Exception as e:
        print(f"Could not load audio: {e}")

    # Prepare the function call message with local file
    messages = process_function_call_message(audio_path)

    try:
        # Run the model using the local API
        response = run_model_local(messages)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error calling local API: {e}")
        print("Make sure the local server is running on http://localhost:8080/v1")