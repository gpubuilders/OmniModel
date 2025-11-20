#!/usr/bin/env python
# coding: utf-8

"""
Simple test to check if the local API is working
"""

import requests
import json

def test_local_api():
    # API endpoint
    api_url = "http://localhost:8080/v1/chat/completions"
    
    # Simple text-only message
    payload = {
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
    
    # Make the request
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print("API is working correctly!")
            print(f"Response content: {result['choices'][0]['message']['content']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    test_local_api()