#!/usr/bin/env python
# coding: utf-8

"""
Test video support with smaller snippet
"""

import requests
import json
import base64
import subprocess
import os

def create_video_snippet(input_path, output_path, duration=5):
    """Create a small snippet of the video to reduce size"""
    try:
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-t", str(duration),  # Duration in seconds
            "-c:v", "libx264",    # Video codec
            "-c:a", "aac",        # Audio codec
            "-movflags", "faststart",  # Optimize for streaming
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Successfully created video snippet: {output_path}")
            return True
        else:
            print(f"Error creating video snippet: {result.stderr}")
            return False
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to create video snippets.")
        return False

def test_small_video():
    # Create a small snippet of the video
    input_video = "assets/draw.mp4"
    small_video = "assets/draw_small.mp4"
    
    if os.path.exists(small_video):
        os.remove(small_video)  # Remove old small video if exists
    
    if not create_video_snippet(input_video, small_video, duration=2):
        print("Could not create video snippet. Using original file.")
        small_video = input_video
    
    # Get the size of the video
    size_mb = os.path.getsize(small_video) / (1024 * 1024)
    print(f"Video size: {size_mb:.2f} MB")
    
    # API endpoint
    api_url = "http://localhost:8080/v1/chat/completions"
    
    # Try with the small video file
    try:
        with open(small_video, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        
        print(f"Encoded video size: {len(video_base64)} characters")
        
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
    
    print("--- Testing Video Support with Small Snippet ---")
    try:
        response = requests.post(api_url, headers=headers, json=video_payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print("Small video support is working!")
            print(f"Response content: {result['choices'][0]['message']['content'][:100]}...")
        else:
            print(f"Small video support error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Small video request exception: {e}")

if __name__ == "__main__":
    test_small_video()