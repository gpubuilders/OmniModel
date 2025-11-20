#!/usr/bin/env python
# coding: utf-8

"""
Test to understand the proper format for multimodal requests to the local API
"""

import requests
import json
import base64

def test_multimodal_format():
    # API endpoint
    api_url = "http://localhost:8080/v1/chat/completions"
    
    # Check if the model info shows multimodal support
    models_url = "http://localhost:8080/v1/models"
    try:
        response = requests.get(models_url)
        print(f"Models endpoint response: {response.text}")
    except:
        print("Could not access models endpoint")
    
    # Try different message formats for multimodal input
    test_cases = [
        # Basic text message
        {
            "name": "Basic text",
            "payload": {
                "model": "Qwen3-Omni-10k",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, how are you?"
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 100
            }
        },
        # Text with image URL (if the image is accessible)
        {
            "name": "Text with image URL",
            "payload": {
                "model": "Qwen3-Omni-10k",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/2621.jpg"}
                            },
                            {
                                "type": "text",
                                "text": "What do you see in this image?"
                            }
                        ]
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 100
            }
        },
        # Text with image as base64 (local file)
        {
            "name": "Text with local image base64",
            "payload": {
                # We'll update this with actual base64 data
            }
        }
    ]
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Prepare the base64 image test case
    try:
        with open("assets/2621.jpg", "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        test_cases[2]["payload"] = {
            "model": "Qwen3-Omni-10k",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        },
                        {
                            "type": "text",
                            "text": "What do you see in this image?"
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 100
        }
    except Exception as e:
        print(f"Could not prepare base64 image test: {e}")
        # Skip this test
        test_cases.pop(2)
    
    # Run tests
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")
        try:
            response = requests.post(api_url, headers=headers, json=test_case['payload'])
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print("Success!")
                print(f"Response: {result['choices'][0]['message']['content'][:100]}...")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == "__main__":
    test_multimodal_format()