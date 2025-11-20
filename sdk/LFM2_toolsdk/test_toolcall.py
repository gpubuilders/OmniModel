"""
Tool Calling API Test Suite

Intent: Test a local LLM server's tool calling capabilities including:
1. Basic tool invocation
2. Tool visibility (model awareness of available tools)
3. Sequential multi-tool calls
4. Multi-turn conversations with tool results

Server: localhost:8080/v1 (OpenAI-compatible with custom tool call format)
"""

import requests
import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


BASE_URL = "http://localhost:8080/v1"
MODEL_NAME = "LFM2-8B-A1B-BF16-cuda"


@dataclass
class APIResponse:
    """Structured API response"""
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    raw_data: Optional[Dict] = None


TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    },
    {
        "type": "function",
        "name": "calculate",
        "description": "Do math",
        "parameters": {
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"]
        }
    },
    {
        "type": "function",
        "name": "search_web",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "type": "function",
        "name": "get_time",
        "description": "Get current time",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"]
        }
    },
    {
        "type": "function",
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
    },
    {
        "type": "function",
        "name": "get_stock",
        "description": "Get stock price",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"]
        }
    },
    {
        "type": "function",
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
    },
    {
        "type": "function",
        "name": "create_task",
        "description": "Create a task",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"]
        }
    },
]


def chat(messages: List[Dict], tools: Optional[List[Dict]] = None) -> APIResponse:
    """
    Send chat completion request to local LLM API.
    
    Returns APIResponse with success status and either content or error.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 512
    }
    
    if tools:
        payload["tools"] = tools
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            json=payload,
            timeout=30
        )
        
        data = response.json()
        print(f"\n[DEBUG] API Response:\n{json.dumps(data, indent=2)}\n")
        
        if "error" in data:
            return APIResponse(
                success=False,
                error=f"API Error {data['error'].get('code', 'unknown')}: {data['error'].get('message', 'no message')}",
                raw_data=data
            )
        
        if "choices" not in data or len(data["choices"]) == 0:
            return APIResponse(
                success=False,
                error="Response missing 'choices' field or choices is empty",
                raw_data=data
            )
        
        content = data["choices"][0]["message"]["content"]
        return APIResponse(success=True, content=content, raw_data=data)
        
    except requests.exceptions.RequestException as e:
        return APIResponse(success=False, error=f"Request failed: {str(e)}")
    except json.JSONDecodeError as e:
        return APIResponse(success=False, error=f"JSON decode failed: {str(e)}")
    except Exception as e:
        return APIResponse(success=False, error=f"Unexpected error: {str(e)}")


def extract_tool_calls(response_content: str) -> List[str]:
    """
    Extract tool calls from response content.
    Expected format: <|tool_call_start|>[func1(), func2()]<|tool_call_end|>
    
    Returns list of tool call strings like ["get_weather(city='Paris')", ...]
    """
    if not response_content or "<|tool_call_start|>" not in response_content:
        return []
    
    start_marker = "<|tool_call_start|>"
    end_marker = "<|tool_call_end|>"
    
    start_idx = response_content.find(start_marker)
    if start_idx == -1:
        return []
    
    start_idx += len(start_marker)
    end_idx = response_content.find(end_marker, start_idx)
    
    if end_idx == -1:
        return []
    
    calls_section = response_content[start_idx:end_idx].strip()
    calls_section = calls_section.strip("[]").strip()
    
    if not calls_section:
        return []
    
    # Extract function call patterns: function_name(...)
    pattern = r'\w+\([^)]*\)'
    return re.findall(pattern, calls_section)


def test_basic_call() -> Tuple[bool, str]:
    """Test 1: Basic single tool call"""
    print("\n" + "="*60)
    print("TEST 1: Basic Tool Call")
    print("="*60)
    
    tools = [TOOLS[0]]  # Just get_weather
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    
    result = chat(messages, tools)
    
    if not result.success:
        return False, f"API call failed: {result.error}"
    
    calls = extract_tool_calls(result.content)
    
    print(f"Response preview: {result.content[:200]}...")
    print(f"Extracted calls: {calls}")
    print(f"Status: {'PASS' if len(calls) > 0 else 'FAIL'}")
    
    return len(calls) > 0, "Basic call successful" if len(calls) > 0 else "No tool calls extracted"


def test_tool_visibility(max_tools: int = 8) -> Tuple[int, str]:
    """Test 2: Tool visibility - can model see all provided tools?"""
    print("\n" + "="*60)
    print("TEST 2: Tool Visibility")
    print("="*60)
    
    max_visible = 0
    
    for n in range(1, max_tools + 1):
        tools = TOOLS[:n]
        tool_names = [t["name"] for t in tools]
        
        prompt = f"List all {n} available tools by name"
        result = chat([{"role": "user", "content": prompt}], tools)
        
        if not result.success:
            print(f"  {n} tools: FAIL - {result.error}")
            break
        
        mentioned_count = sum(1 for name in tool_names if name in result.content)
        status = "PASS" if mentioned_count == n else "FAIL"
        
        print(f"  {n} tools: {status} ({mentioned_count}/{n} visible)")
        
        if mentioned_count == n:
            max_visible = n
        else:
            break
    
    return max_visible, f"Model can see up to {max_visible} tools"


def test_sequential_calls() -> Tuple[int, str]:
    """Test 3: Sequential multi-tool calls in single turn"""
    print("\n" + "="*60)
    print("TEST 3: Sequential Multi-Tool Calls")
    print("="*60)
    
    tools = TOOLS[:3]  # get_weather, calculate, search_web
    
    test_cases = [
        (1, "What's the weather in Paris?"),
        (2, "Get weather in Paris and calculate 5+3"),
        (3, "Get weather in Paris, calculate 10*2, and search for 'AI news'"),
    ]
    
    max_achieved = 0
    
    for expected_count, prompt in test_cases:
        print(f"\n  Expected {expected_count} calls: {prompt}")
        
        result = chat([{"role": "user", "content": prompt}], tools)
        
        if not result.success:
            print(f"    FAIL - {result.error}")
            break
        
        calls = extract_tool_calls(result.content)
        actual_count = len(calls)
        status = "PASS" if actual_count >= expected_count else "FAIL"
        
        print(f"    {status} - Got {actual_count} calls: {calls}")
        
        if actual_count >= expected_count:
            max_achieved = expected_count
        else:
            break
    
    return max_achieved, f"Model can handle {max_achieved} sequential calls"


def test_multi_turn() -> Tuple[bool, str]:
    """Test 4: Multi-turn conversation with tool results"""
    print("\n" + "="*60)
    print("TEST 4: Multi-Turn Conversation")
    print("="*60)
    
    tools = TOOLS[:2]  # get_weather, calculate
    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    
    print("\n  Turn 1: Initial query")
    result = chat(messages, tools)
    
    if not result.success:
        return False, f"Turn 1 failed: {result.error}"
    
    calls = extract_tool_calls(result.content)
    print(f"    Got {len(calls)} calls: {calls}")
    
    if len(calls) == 0:
        return False, "No tool calls in turn 1"
    
    # Simulate tool result and continue conversation
    messages.append({"role": "assistant", "content": result.content})
    messages.append({
        "role": "tool",
        "content": json.dumps([{"temp": 22, "condition": "sunny"}])
    })
    messages.append({"role": "user", "content": "Now calculate 15 * 4"})
    
    print("\n  Turn 2: Follow-up query")
    result = chat(messages, tools)
    
    if not result.success:
        return False, f"Turn 2 failed: {result.error}"
    
    calls = extract_tool_calls(result.content)
    print(f"    Got {len(calls)} calls: {calls}")
    
    success = len(calls) > 0
    message = "Multi-turn successful" if success else "No tool calls in turn 2"
    
    print(f"    Status: {'PASS' if success else 'FAIL'}")
    
    return success, message


def main():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("TOOL CALLING API TEST SUITE")
    print(f"Server: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print("="*60)
    
    results = {}
    
    # Test 1: Basic call
    success, message = test_basic_call()
    results["basic"] = (success, message)
    
    if not success:
        print("\n" + "="*60)
        print("CRITICAL FAILURE: Basic test failed")
        print(f"Reason: {message}")
        print("="*60)
        return
    
    # Test 2: Visibility
    max_visible, message = test_tool_visibility(len(TOOLS))
    results["visibility"] = (max_visible, message)
    
    # Test 3: Sequential
    max_sequential, message = test_sequential_calls()
    results["sequential"] = (max_sequential, message)
    
    # Test 4: Multi-turn
    success, message = test_multi_turn()
    results["multi_turn"] = (success, message)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Basic Call:        {'PASS' if results['basic'][0] else 'FAIL'}")
    print(f"Tool Visibility:   {results['visibility'][0]}/{len(TOOLS)} tools")
    print(f"Sequential Calls:  {results['sequential'][0]} max")
    print(f"Multi-Turn:        {'PASS' if results['multi_turn'][0] else 'FAIL'}")
    print("="*60)
    
    # Detailed messages
    print("\nDetails:")
    for test_name, (result, message) in results.items():
        print(f"  {test_name}: {message}")


if __name__ == "__main__":
    main()
