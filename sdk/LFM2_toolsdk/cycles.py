"""
Tool Calling Test Framework - Clean Architecture

Philosophy: Every tool call gets executed (mocked) and the model's response is evaluated.

Flow:
1. User query → Model generates tool call
2. Execute tool (mock) → Get result
3. Feed result back → Model generates natural language response
4. Evaluate the complete cycle
"""

import requests
import json
from typing import List, Dict, Optional, Tuple
import re


BASE_URL = "http://localhost:8080/v1"
MODEL_NAME = "LFM2-8B-A1B-BF16-cuda"


def call_api(messages: List[Dict], tools: Optional[List[Dict]] = None) -> str:
    """Raw API call - returns response content"""
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


def extract_tool_calls(content: str) -> List[str]:
    """Extract tool calls from model response"""
    if "<|tool_call_start|>" not in content:
        return []
    
    start = content.find("<|tool_call_start|>") + len("<|tool_call_start|>")
    end = content.find("<|tool_call_end|>")
    if end == -1:
        return []
    
    calls_str = content[start:end].strip().strip("[]")
    if not calls_str:
        return []
    
    return re.findall(r'\w+\([^)]*\)', calls_str)


def mock_tool_execution(tool_call: str) -> str:
    """Mock tool execution - returns JSON string"""
    if "get_weather" in tool_call:
        city = tool_call.split('"')[1] if '"' in tool_call else "Unknown"
        return json.dumps({
            "city": city,
            "temperature": 22,
            "condition": "sunny",
            "humidity": 65
        })
    
    elif "get_stock" in tool_call:
        symbol = tool_call.split('"')[1] if '"' in tool_call else "UNKNOWN"
        return json.dumps({
            "symbol": symbol,
            "price": 150.25,
            "change": 2.5,
            "change_percent": 1.69
        })
    
    elif "search_web" in tool_call:
        return json.dumps({
            "results": [
                "AI breakthrough in language models",
                "New machine learning framework released",
                "Tech company announces AI product"
            ]
        })
    
    elif "calculate" in tool_call:
        return json.dumps({"result": 42})
    
    return json.dumps({"error": "Unknown tool"})


def run_tool_cycle(
    user_query: str,
    tools: List[Dict],
    conversation: Optional[List[Dict]] = None,
    verbose: bool = True
) -> Tuple[str, List[str], str]:
    """
    Run a complete tool calling cycle:
    1. User query → Tool call
    2. Execute tool → Get result  
    3. Result → Natural language response
    
    Returns: (tool_call_response, extracted_calls, final_response)
    """
    if conversation is None:
        conversation = []
    
    # Step 1: User query
    conversation.append({"role": "user", "content": user_query})
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"USER: {user_query}")
        print(f"{'='*60}")
    
    # Step 2: Model generates tool call
    tool_call_response = call_api(conversation, tools)
    calls = extract_tool_calls(tool_call_response)
    
    if verbose:
        print(f"MODEL TOOL CALL: {calls if calls else 'No tool call'}")
    
    if not calls:
        # No tool call, model answered directly
        if verbose:
            print(f"MODEL RESPONSE: {tool_call_response}")
        return tool_call_response, [], tool_call_response
    
    # Step 3: Execute tool(s)
    conversation.append({"role": "assistant", "content": tool_call_response})
    
    for call in calls:
        tool_result = mock_tool_execution(call)
        if verbose:
            print(f"TOOL RESULT: {tool_result}")
        conversation.append({"role": "tool", "content": tool_result})
    
    # Step 4: Model interprets result and responds
    final_response = call_api(conversation, tools)
    
    if verbose:
        print(f"MODEL FINAL: {final_response}")
    
    conversation.append({"role": "assistant", "content": final_response})
    
    return tool_call_response, calls, final_response


# Define tools
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
}

STOCK_TOOL = {
    "type": "function",
    "function": {
        "name": "get_stock",
        "description": "Get stock price for a symbol",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"]
        }
    }
}

SEARCH_TOOL = {
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
}


def test_single_tool_cycle():
    """Test: Single tool call with complete cycle"""
    print("\n" + "="*60)
    print("TEST: Single Tool Call Cycle")
    print("="*60)
    
    _, calls, final = run_tool_cycle(
        "What's the weather in Tokyo?",
        [WEATHER_TOOL]
    )
    
    success = len(calls) > 0 and "Tokyo" in final
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}")
    return success


def test_multi_turn_cycle():
    """Test: Multi-turn conversation with tool calls"""
    print("\n" + "="*60)
    print("TEST: Multi-Turn Tool Cycle")
    print("="*60)
    
    conversation = []
    tools = [WEATHER_TOOL, STOCK_TOOL]
    
    # Turn 1: Weather
    _, calls1, final1 = run_tool_cycle(
        "What's the weather in Paris?",
        tools,
        conversation
    )
    
    # Turn 2: Stock (different tool)
    _, calls2, final2 = run_tool_cycle(
        "Now get the stock price for AAPL",
        tools,
        conversation
    )
    
    # Turn 3: Use context (no tool needed)
    _, calls3, final3 = run_tool_cycle(
        "Which one should I care about more?",
        tools,
        conversation
    )
    
    success = (
        len(calls1) > 0 and
        len(calls2) > 0 and
        len(calls3) == 0 and  # Should answer from context
        "Paris" in final1 and
        "AAPL" in final2
    )
    
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}")
    return success


def test_sequential_tools_cycle():
    """Test: Multiple tools in one turn"""
    print("\n" + "="*60)
    print("TEST: Sequential Tools in Single Turn")
    print("="*60)
    
    tools = [WEATHER_TOOL, SEARCH_TOOL]
    
    _, calls, final = run_tool_cycle(
        "Get weather in London and search the web for 'AI news'",
        tools
    )
    
    success = (
        len(calls) >= 2 and
        any("weather" in c.lower() for c in calls) and
        any("search" in c.lower() for c in calls)
    )
    
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}")
    return success


def test_tool_result_usage():
    """Test: Model actually uses tool results"""
    print("\n" + "="*60)
    print("TEST: Model Uses Tool Results")
    print("="*60)
    
    conversation = []
    
    # Get weather for Tokyo
    _, calls1, final1 = run_tool_cycle(
        "What's the weather in Tokyo?",
        [WEATHER_TOOL],
        conversation
    )
    
    # Get weather for London  
    _, calls2, final2 = run_tool_cycle(
        "What about London?",
        [WEATHER_TOOL],
        conversation
    )
    
    # Ask comparison - should use both previous results
    _, calls3, final3 = run_tool_cycle(
        "Which city is warmer?",
        [WEATHER_TOOL],
        conversation
    )
    
    success = (
        len(calls1) > 0 and
        len(calls2) > 0 and
        len(calls3) == 0 and  # No new tool call needed
        ("Tokyo" in final3 or "London" in final3) and
        "warmer" in final3.lower()
    )
    
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}")
    return success


def main():
    """Run all tests"""
    print("="*60)
    print("TOOL CALLING TEST SUITE - CLEAN ARCHITECTURE")
    print("="*60)
    
    results = {
        "Single Tool Cycle": test_single_tool_cycle(),
        "Multi-Turn Cycle": test_multi_turn_cycle(),
        "Sequential Tools": test_sequential_tools_cycle(),
        "Tool Result Usage": test_tool_result_usage(),
    }
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for test_name, passed in results.items():
        print(f"{test_name}: {'✓ PASS' if passed else '✗ FAIL'}")
    print("="*60)
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nPassed: {passed}/{total}")


if __name__ == "__main__":
    main()
