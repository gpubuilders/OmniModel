"""
Tool Calling API Test Suite - Enhanced with Full Response Logging

Shows complete server responses to debug issues
"""

import requests
import json
from typing import List, Dict, Tuple
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


def chat(messages: List[Dict], tools: List[Dict] = None, verbose: bool = True) -> Tuple[str, Dict]:
    """
    Send chat completion request
    Returns: (content, full_response_data)
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 512
    }
    if tools:
        payload["tools"] = tools
    
    if verbose:
        print("\n" + "-"*60)
        print("REQUEST:")
        print("-"*60)
        print(f"Messages ({len(messages)} total):")
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            content_preview = content[:100] + "..." if len(content) > 100 else content
            print(f"  [{i}] {role}: {content_preview}")
        if tools:
            print(f"Tools: {len(tools)} provided")
    
    response = requests.post(f"{BASE_URL}/chat/completions", json=payload)
    data = response.json()
    
    if verbose:
        print("\n" + "-"*60)
        print("RESPONSE:")
        print("-"*60)
        print(json.dumps(data, indent=2))
        print("-"*60)
    
    if "error" in data:
        raise Exception(f"API Error: {data['error']['message']}")
    
    content = data["choices"][0]["message"]["content"]
    return content, data


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
    
    response, _ = chat(messages, tools)
    calls = extract_tool_calls(response)
    
    print(f"\nExtracted calls: {calls}")
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
        
        response, _ = chat(
            [{"role": "user", "content": f"List all {n} available tools"}],
            tools,
            verbose=False  # Too noisy for this test
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
        print(f"\n--- Testing {expected} call(s): {prompt} ---")
        response, _ = chat([{"role": "user", "content": prompt}], tools)
        calls = extract_tool_calls(response)
        
        status = "✓" if len(calls) >= expected else "✗"
        print(f"\nResult: {status} Expected {expected}, got {len(calls)}")
        print(f"Calls: {calls}")
        
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
    print("\n" + "="*60)
    print("TURN 1: Initial query")
    print("="*60)
    response, data1 = chat(messages, tools)
    calls = extract_tool_calls(response)
    
    print(f"\nExtracted calls: {calls}")
    
    if not calls:
        print("\n✗ FAIL: No tool calls in turn 1")
        return False
    
    # Turn 2 - simulate tool response and continue
    print("\n" + "="*60)
    print("TURN 2: After tool execution")
    print("="*60)
    
    # Add assistant response
    messages.append({"role": "assistant", "content": response})
    
    # Add tool result
    tool_result = json.dumps([{"temp": 22, "condition": "sunny"}])
    messages.append({
        "role": "tool",
        "content": tool_result
    })
    
    # Add new user query
    messages.append({"role": "user", "content": "Now calculate 15 * 4"})
    
    print(f"\nFull conversation state before Turn 2:")
    for i, msg in enumerate(messages):
        print(f"  [{i}] {msg['role']}: {msg['content'][:80]}...")
    
    response, data2 = chat(messages, tools)
    calls = extract_tool_calls(response)
    
    print(f"\nExtracted calls: {calls}")
    
    if len(calls) > 0:
        print("\n✓ PASS: Tool call found in turn 2")
        return True
    else:
        print("\n✗ FAIL: No tool calls in turn 2")
        print("\nPossible reasons:")
        print("  1. Model interpreted tool result and answered in natural language")
        print("  2. Model doesn't maintain tool calling context across turns")
        print("  3. Tool role message format may be incorrect")
        return False


def test_multi_turn_detailed():
    """
    Additional test: Try different tool response formats
    """
    print("\n" + "="*60)
    print("TEST 4B: Multi-Turn with Different Tool Response Formats")
    print("="*60)
    
    tools = TOOLS[:2]
    
    # Try format 1: JSON array string
    print("\n--- Format 1: Tool result as JSON array string ---")
    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    response, _ = chat(messages, tools, verbose=False)
    calls = extract_tool_calls(response)
    
    if calls:
        messages.append({"role": "assistant", "content": response})
        messages.append({
            "role": "tool",
            "content": '[{"temp": 22, "condition": "sunny"}]'
        })
        messages.append({"role": "user", "content": "Calculate 15 * 4"})
        
        response, _ = chat(messages, tools)
        calls2 = extract_tool_calls(response)
        print(f"Result: {'✓' if calls2 else '✗'} - Got {len(calls2)} calls")
    
    # Try format 2: Plain string response
    print("\n--- Format 2: Tool result as plain string ---")
    messages = [{"role": "user", "content": "What's the weather in London?"}]
    response, _ = chat(messages, tools, verbose=False)
    calls = extract_tool_calls(response)
    
    if calls:
        messages.append({"role": "assistant", "content": response})
        messages.append({
            "role": "tool",
            "content": "Temperature is 22°C, sunny"
        })
        messages.append({"role": "user", "content": "Calculate 20 * 3"})
        
        response, _ = chat(messages, tools)
        calls2 = extract_tool_calls(response)
        print(f"Result: {'✓' if calls2 else '✗'} - Got {len(calls2)} calls")
    
    # Try format 3: With special markers like in chat template
    print("\n--- Format 3: Tool result with response markers ---")
    messages = [{"role": "user", "content": "What's the weather in Berlin?"}]
    response, _ = chat(messages, tools, verbose=False)
    calls = extract_tool_calls(response)
    
    if calls:
        messages.append({"role": "assistant", "content": response})
        messages.append({
            "role": "tool",
            "content": '<|tool_response_start|>[{"temp": 22, "condition": "sunny"}]<|tool_response_end|>'
        })
        messages.append({"role": "user", "content": "Calculate 25 * 2"})
        
        response, _ = chat(messages, tools)
        calls2 = extract_tool_calls(response)
        print(f"Result: {'✓' if calls2 else '✗'} - Got {len(calls2)} calls")


def main():
    """Run all tests"""
    print("="*60)
    print("TOOL CALLING API TEST SUITE - VERBOSE MODE")
    print(f"Server: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print("Format: OpenAI nested (type + function object)")
    print("="*60)
    
    try:
        # Test 1: Basic call
        if not test_basic_call():
            print("\n✗ CRITICAL FAILURE: Basic test failed")
            return
        
        # Test 2: Visibility (brief mode)
        vis = test_tool_visibility(len(TOOLS))
        
        # Test 3: Sequential
        seq = test_sequential_calls()
        
        # Test 4: Multi-turn (detailed)
        multi = test_multi_turn()
        
        # Test 4B: Additional multi-turn testing
        test_multi_turn_detailed()
        
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
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
