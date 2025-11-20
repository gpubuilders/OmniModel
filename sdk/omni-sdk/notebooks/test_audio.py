#!/usr/bin/env python
# coding: utf-8

"""
Test to check audio support specifically with the local API
"""

import requests
import json
import base64

def test_audio_support():
    # API endpoint
    api_url = "http://localhost:8080/v1/chat/completions"
    
    # Try audio with URL
    audio_url_payload = {
        "model": "Qwen3-Omni-10k",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/caption1.mp3"}
                    },
                    {
                        "type": "text",
                        "text": "Give the detailed description of the audio."
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 100
    }
    
    # Try audio with base64 (local file)
    try:
        with open("assets/caption1.mp3", "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        
        audio_base64_payload = {
            "model": "Qwen3-Omni-10k",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/mp3;base64,{audio_base64}"}
                        },
                        {
                            "type": "text",
                            "text": "Give the detailed description of the audio."
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 100
        }
    except Exception as e:
        print(f"Could not prepare base64 audio test: {e}")
        return
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("--- Testing Audio with URL ---")
    try:
        response = requests.post(api_url, headers=headers, json=audio_url_payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("Audio with URL Success!")
            print(f"Response: {result['choices'][0]['message']['content'][:100]}...")
        else:
            print(f"Audio with URL Error: {response.text}")
    except Exception as e:
        print(f"Audio with URL Exception: {e}")
    
    print("\n--- Testing Audio with base64 ---")
    try:
        response = requests.post(api_url, headers=headers, json=audio_base64_payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("Audio with base64 Success!")
            print(f"Response: {result['choices'][0]['message']['content'][:100]}...")
        else:
            print(f"Audio with base64 Error: {response.text}")
    except Exception as e:
        print(f"Audio with base64 Exception: {e}")

if __name__ == "__main__":
    test_audio_support()