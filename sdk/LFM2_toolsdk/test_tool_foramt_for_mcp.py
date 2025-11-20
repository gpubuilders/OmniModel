"""
DEFINITIVE TOOL FORMAT TEST
Tests if LFM2 actually calls tools with the correct nested format
"""
import requests
import json

LLM_URL = "http://localhost:8080/v1"
MODEL = "LFM2-1.2B-Tool-Q4_K_M-cuda"

# ✅ CORRECT NESTED FORMAT (as per your spec)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

def test_tool_calling():
    """Test if model actually calls tools"""
    
    print("="*60)
    print("TOOL FORMAT TEST - LFM2")
    print("="*60)
    
    test_cases = [
        {
            "name": "Weather Query",
            "query": "What's the weather in Paris? Use the get_weather tool.",
            "expected_tool": "get_weather"
        },
        {
            "name": "Math Query", 
            "query": "Calculate 15 + 27 using the calculate tool.",
            "expected_tool": "calculate"
        },
        {
            "name": "Explicit Instruction",
            "query": "YOU MUST CALL THE get_weather TOOL. Get weather for Tokyo.",
            "expected_tool": "get_weather"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {test['name']}")
        print(f"{'='*60}")
        print(f"Query: {test['query']}")
        
        messages = [{"role": "user", "content": test["query"]}]
        
        # Call LLM with tools
        response = requests.post(
            f"{LLM_URL}/chat/completions",
            json={
                "model": MODEL,
                "messages": messages,
                "tools": TOOLS,
                "temperature": 0,
                "max_tokens": 256
            }
        )
        
        if response.status_code != 200:
            print(f"✗ HTTP Error: {response.status_code}")
            print(f"  Response: {response.text}")
            continue
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        print(f"\nRaw Response:")
        print(content)
        
        # Check for tool call markers
        if "<|tool_call_start|>" in content and "<|tool_call_end|>" in content:
            print(f"\n✓ Tool call detected!")
            
            # Extract tool call
            start = content.find("<|tool_call_start|>") + len("<|tool_call_start|>")
            end = content.find("<|tool_call_end|>")
            tool_str = content[start:end].strip()
            
            print(f"  Tool call: {tool_str}")
            
            # Check if correct tool
            if test["expected_tool"] in tool_str:
                print(f"  ✓ Correct tool: {test['expected_tool']}")
            else:
                print(f"  ✗ Wrong tool (expected {test['expected_tool']})")
        else:
            print(f"\n✗ NO TOOL CALL - Model responded without calling tools")
            print(f"  This means the model ignored the tools")
    
    # Test 4: Show tool format being sent
    print(f"\n{'='*60}")
    print("VERIFY: Tool Format Being Sent")
    print(f"{'='*60}")
    print(json.dumps(TOOLS[0], indent=2))

if __name__ == "__main__":
    test_tool_calling()
