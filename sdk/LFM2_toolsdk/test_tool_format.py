"""
Deep diagnostic: The server gives contradictory errors
- With "type":"function" → "Missing tool function"
- Without "type" → "Missing tool type"

This suggests the server expects BOTH a "type" field AND a "function" field.
Possible schemas to try:
1. Nested structure: {"type": "function", "function": {...}}
2. Different type value: {"type": "tool", ...} or {"type": "action", ...}
3. Function as separate field: {"type": "function", "function": {"name": ..., "description": ...}}
"""

import requests
import json
from typing import List, Dict


BASE_URL = "http://localhost:8080/v1"
MODEL_NAME = "LFM2-8B-A1B-BF16-cuda"


def test_format(format_name: str, tools_payload, messages: List[Dict]):
    """Test a specific tool format"""
    print(f"\n{'='*60}")
    print(f"Testing: {format_name}")
    print(f"{'='*60}")
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 512,
        "tools": tools_payload
    }
    
    print(f"Tools structure:")
    print(json.dumps(tools_payload, indent=2)[:300] + "...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            json=payload,
            timeout=30
        )
        
        data = response.json()
        
        if "error" in data:
            error_msg = data['error']['message']
            print(f"❌ FAILED")
            # Extract just the key part of error
            if "Missing tool" in error_msg:
                print(f"   Error: {error_msg.split(';')[0]}")
            else:
                print(f"   Error: {error_msg[:150]}")
            return False
        
        if "choices" in data:
            content = data["choices"][0]["message"]["content"]
            print(f"✅ SUCCESS!")
            print(f"Response: {content[:150]}...")
            
            if "<|tool_call_start|>" in content:
                start = content.find("<|tool_call_start|>") + len("<|tool_call_start|>")
                end = content.find("<|tool_call_end|>")
                if end != -1:
                    calls = content[start:end]
                    print(f"Tool calls found: {calls}")
            return True
        
        print(f"❌ Unexpected response format")
        return False
        
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        return False


def main():
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    
    print("="*60)
    print("LFM2 Tool Format Deep Diagnostic")
    print("="*60)
    print("\nHypothesis: Server expects nested structure with both")
    print("'type' field AND separate 'function' object")
    print("="*60)
    
    # Format 1: OpenAI's actual nested format (newer spec)
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
    test_format(
        "Format 1: OpenAI nested structure (type + function object)",
        [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }],
        messages
    )
    
    # Format 2: Type as "tool" instead of "function"
    test_format(
        "Format 2: type='tool' with function object",
        [{
            "type": "tool",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }],
        messages
    )
    
    # Format 3: Flat structure with both type and function fields
    test_format(
        "Format 3: Flat with type='function' and function=name",
        [{
            "type": "function",
            "function": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }],
        messages
    )
    
    # Format 4: Maybe they use "callable" or "method"?
    test_format(
        "Format 4: type='callable'",
        [{
            "type": "callable",
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }],
        messages
    )
    
    # Format 5: String type values
    test_format(
        "Format 5: type='string' (weird but let's try)",
        [{
            "type": "string",
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }],
        messages
    )
    
    # Format 6: Maybe "function" should be at top level differently?
    test_format(
        "Format 6: function as top-level wrapper",
        [{
            "function": {
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }],
        messages
    )
    
    # Format 7: Check if it needs strict JSON schema compliance
    test_format(
        "Format 7: Full JSON schema format",
        [{
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        }],
        messages
    )
    
    # Format 8: Maybe no parameters wrapper?
    test_format(
        "Format 8: Direct properties (no parameters wrapper)",
        [{
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }],
        messages
    )
    
    print("\n" + "="*60)
    print("If all failed, check:")
    print("1. Server source code for tool parsing logic")
    print("2. Server logs for detailed stack trace")
    print("3. Try calling /v1/models endpoint to check capabilities")
    print("="*60)


if __name__ == "__main__":
    main()
