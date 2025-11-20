#!/usr/bin/env python
# coding: utf-8

"""
Video support smoke test for local API
"""

import requests
import json
import base64

def test_video_support():
    # API endpoint
    api_url = "http://localhost:8080/v1/chat/completions"
    
    # Try with a video file
    try:
        with open("assets/draw.mp4", "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        
        video_payload = {
            "model": "Qwen3-Omni-10k",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url", 
                            "video_url": {
                                "url": f"data:video/mp4;base64,{video_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "What can you see in this video?"
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 100
        }
    except Exception as e:
        print(f"Could not prepare video test: {e}")
        return
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("--- Testing Video Support ---")
    try:
        response = requests.post(api_url, headers=headers, json=video_payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print("Video support is working!")
            print(f"Response content: {result['choices'][0]['message']['content'][:100]}...")
        else:
            print(f"Video support error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Video request exception: {e}")

def test_video_with_different_format():
    # API endpoint
    api_url = "http://localhost:8080/v1/chat/completions"
    
    # Try with a video file - different format
    try:
        with open("assets/draw.mp4", "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        
        # Try different payload format
        video_payload = {
            "model": "Qwen3-Omni-10k",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"data:video/mp4;base64,{video_base64}"
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 100
        }
    except Exception as e:
        print(f"Could not prepare alternative video test: {e}")
        return
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("\n--- Testing Video Support with Alternative Format ---")
    try:
        response = requests.post(api_url, headers=headers, json=video_payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print("Video support is working with alternative format!")
            print(f"Response content: {result['choices'][0]['message']['content'][:100]}...")
        else:
            print(f"Video support error with alternative format: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Alternative video request exception: {e}")

if __name__ == "__main__":
    test_video_support()
    test_video_with_different_format()