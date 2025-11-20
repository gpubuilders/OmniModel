"""
Tool Calling API Test Suite - Working Version

Uses the correct OpenAI nested format:
{
  "type": "function",
  "function": {
    "name": "...",
    "description": "...",
    "parameters": {...}
  }
}
"""

import requests
import json
from typing import List, Dict
import re


BASE_URL = "http://localhost:8080/v1"
MODEL_NAME = "LFM2-8B-A1B-BF16-cuda"


# Tools in correct nested format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Do math",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
                "required": ["timezone"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "Translate text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target": {"type": "string"}
                },
                "required": ["text", "target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock",
            "description": "Get stock price",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"}
                },
                "required": ["to", "subject"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create a task",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"]
            }
        }
    },
]


def chat(messages: List[Dict], tools: List[Dict] = None) -> str:
    """Send chat completion request"""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 512
    }
    if tools:
        payload["tools"] = tools
    
    response = requests.post(f"{BASE_URL}/chat/completions", json=payload)
    data = response.json()
    
    if "error" in data:
        raise Exception(f"API Error: {data['error']['message']}")
    
    return data["choices"][0]["message"]["content"]


def extract_tool_calls(response: str) -> List[str]:
    """Extract tool calls from response"""
    if "<|tool_call_start|>" not in response:
        return []
    
    start = response.find("<|tool_call_start|>") + len("<|tool_call_start|>")
    end = response.find("<|tool_call_end|>")
    if end == -1:
        return []
    
    calls_str = response[start:end].strip().strip("[]")
    if not calls_str:
        return []
    
    return re.findall(r'\w+\([^)]*\)', calls_str)


def test_basic_call():
    """Test 1: Basic single tool call"""
    print("\n" + "="*60)
    print("TEST 1: Basic Tool Call")
    print("="*60)
    
    tools = [TOOLS[0]]
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    response = chat(messages, tools)
    calls = extract_tool_calls(response)
    
    print(f"Response: {response[:200]}")
    print(f"Calls: {calls}")
    print(f"Status: {'✓ PASS' if len(calls) > 0 else '✗ FAIL'}")
    
    return len(calls) > 0


def test_tool_visibility(max_tools: int = 8):
    """Test 2: Tool visibility - can model see all provided tools?"""
    print("\n" + "="*60)
    print("TEST 2: Tool Visibility")
    print("="*60)
    
    for n in range(1, max_tools + 1):
        tools = TOOLS[:n]
        tool_names = [t["function"]["name"] for t in tools]
        
        response = chat(
            [{"role": "user", "content": f"List all {n} available tools"}],
            tools
        )
        
        mentioned = sum(1 for name in tool_names if name in response)
        status = "✓" if mentioned == n else "✗"
        
        print(f"{n} tools: {status} ({mentioned}/{n} mentioned)")
        
        if mentioned != n:
            return n - 1
    
    return max_tools


def test_sequential_calls():
    """Test 3: Sequential multi-tool calls in single turn"""
    print("\n" + "="*60)
    print("TEST 3: Sequential Multi-Tool Calls")
    print("="*60)
    
    tools = TOOLS[:3]
    tests = [
        (1, "What's the weather in Paris?"),
        (2, "Get weather in Paris and calculate 5+3"),
        (3, "Get weather in Paris, calculate 10*2, and search for 'AI news'"),
    ]
    
    for expected, prompt in tests:
        response = chat([{"role": "user", "content": prompt}], tools)
        calls = extract_tool_calls(response)
        
        status = "✓" if len(calls) >= expected else "✗"
        print(f"{expected} calls: {status} (got {len(calls)})")
        
        if calls:
            print(f"  {calls}")
        
        if len(calls) < expected:
            return expected - 1
    
    return 3


def test_multi_turn():
    """Test 4: Multi-turn conversation with tool results"""
    print("\n" + "="*60)
    print("TEST 4: Multi-Turn Conversation")
    print("="*60)
    
    tools = TOOLS[:2]
    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    
    # Turn 1
    response = chat(messages, tools)
    calls = extract_tool_calls(response)
    print(f"Turn 1: {len(calls)} calls → {calls}")
    
    if not calls:
        print("Status: ✗ FAIL (no tool calls)")
        return False
    
    # Turn 2 - simulate tool response and continue
    messages.append({"role": "assistant", "content": response})
    messages.append({
        "role": "tool",
        "content": json.dumps([{"temp": 22, "condition": "sunny"}])
    })
    messages.append({"role": "user", "content": "Now calculate 15 * 4"})
    
    response = chat(messages, tools)
    calls = extract_tool_calls(response)
    print(f"Turn 2: {len(calls)} calls → {calls}")
    
    status = len(calls) > 0
    print(f"Status: {'✓ PASS' if status else '✗ FAIL'}")
    
    return status


def main():
    """Run all tests"""
    print("="*60)
    print("TOOL CALLING API TEST SUITE")
    print(f"Server: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print("Format: OpenAI nested (type + function object)")
    print("="*60)
    
    try:
        # Test 1: Basic call
        if not test_basic_call():
            print("\n✗ CRITICAL FAILURE: Basic test failed")
            return
        
        # Test 2: Visibility
        vis = test_tool_visibility(len(TOOLS))
        
        # Test 3: Sequential
        seq = test_sequential_calls()
        
        # Test 4: Multi-turn
        multi = test_multi_turn()
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Tool Visibility: {vis}/{len(TOOLS)} tools")
        print(f"Sequential Calls: {seq} max")
        print(f"Multi-Turn: {'✓ PASS' if multi else '✗ FAIL'}")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")


if __name__ == "__main__":
    main()
