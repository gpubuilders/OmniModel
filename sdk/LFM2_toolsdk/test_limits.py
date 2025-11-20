"""
Tool Calling Limits - Three-Axis Testing

Axis 1: Single tool, increasing parameters
Axis 2: Multiple tools, simple parameters
Axis 3: Sequential calls, simple tools

Each test runs complete cycles with mocked tool execution.
"""

import requests
import json
from typing import List, Dict, Optional, Tuple
import re


BASE_URL = "http://localhost:8080/v1"
MODEL_NAME = "LFM2-1.2B-Tool-Q4_K_M-cuda"
# MODEL_NAME = "LFM2-8B-A1B-UD-Q3_K_XL-cuda" # fail
# MODEL_NAME = "LFM2-8B-A1B-BF16-cuda"


# ============================================================
# CORE FUNCTIONS
# ============================================================

def call_api(messages: List[Dict], tools: Optional[List[Dict]] = None) -> str:
    """Raw API call"""
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
    """Extract tool calls from response"""
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
    """Mock tool execution - generic response"""
    return json.dumps({"status": "success", "data": "mocked_result"})


def run_cycle(user_query: str, tools: List[Dict], verbose: bool = False) -> Tuple[List[str], str]:
    """
    Run complete tool cycle.
    Returns: (extracted_calls, final_response)
    """
    conversation = [{"role": "user", "content": user_query}]
    
    # Get tool calls
    tool_call_response = call_api(conversation, tools)
    calls = extract_tool_calls(tool_call_response)
    
    if verbose:
        print(f"  Query: {user_query[:60]}...")
        print(f"  Calls: {calls}")
    
    if not calls:
        return [], tool_call_response
    
    # Execute tools
    conversation.append({"role": "assistant", "content": tool_call_response})
    for call in calls:
        result = mock_tool_execution(call)
        conversation.append({"role": "tool", "content": result})
    
    # Get final response
    final_response = call_api(conversation, tools)
    
    if verbose:
        print(f"  Final: {final_response[:60]}...")
    
    return calls, final_response


# ============================================================
# TOOL GENERATORS
# ============================================================

def generate_tool_with_n_params(n_params: int) -> Dict:
    """Generate a tool with N parameters"""
    properties = {}
    required = []
    
    for i in range(n_params):
        param_name = f"param_{i+1}"
        properties[param_name] = {"type": "string"}
        required.append(param_name)
    
    return {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": f"Test tool with {n_params} parameters",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }


def generate_n_simple_tools(n_tools: int) -> List[Dict]:
    """Generate N tools with simple parameters"""
    tools = []
    
    for i in range(n_tools):
        tools.append({
            "type": "function",
            "function": {
                "name": f"tool_{i+1}",
                "description": f"This is tool number {i+1}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"}
                    },
                    "required": ["input"]
                }
            }
        })
    
    return tools


# ============================================================
# AXIS 1: SINGLE TOOL - PARAMETER SCALING
# ============================================================

def test_axis1_parameter_scaling():
    """
    Test: Single tool with increasing parameters
    Find: Max parameters before failure
    """
    print("\n" + "="*60)
    print("AXIS 1: Single Tool - Parameter Scaling")
    print("="*60)
    
    param_counts = [1, 2, 5, 10, 15, 20, 25, 30]
    results = {}
    
    for n_params in param_counts:
        print(f"\n--- Testing {n_params} parameters ---")
        
        tool = generate_tool_with_n_params(n_params)
        
        # Generate query mentioning all params
        param_values = [f"value{i+1}" for i in range(n_params)]
        query = f"Call the test tool with these values: {', '.join(param_values)}"
        
        try:
            calls, final = run_cycle(query, [tool], verbose=True)
            
            # Check if tool was called
            success = len(calls) > 0 and "test_tool" in calls[0]
            
            # Try to count how many params were included
            param_count_in_call = sum(1 for i in range(n_params) if f"param_{i+1}" in calls[0]) if calls else 0
            
            results[n_params] = {
                "success": success,
                "called": len(calls) > 0,
                "params_included": param_count_in_call
            }
            
            status = "✓" if success else "✗"
            print(f"  Result: {status} - Included {param_count_in_call}/{n_params} params")
            
        except Exception as e:
            print(f"  Result: ✗ ERROR - {str(e)[:50]}")
            results[n_params] = {"success": False, "error": str(e)}
            break
    
    # Summary
    print("\n" + "-"*60)
    print("AXIS 1 SUMMARY:")
    for n, result in results.items():
        if result.get("success"):
            print(f"  {n} params: ✓ ({result.get('params_included', 0)}/{n} included)")
        else:
            print(f"  {n} params: ✗")
    
    # Find limit
    max_success = max([n for n, r in results.items() if r.get("success")], default=0)
    print(f"\nLimit: {max_success} parameters")
    print("-"*60)
    
    return results


# ============================================================
# AXIS 2: MULTIPLE TOOLS - SIMPLE PARAMETERS
# ============================================================

def test_axis2_multiple_tools():
    """
    Test: Multiple tools with simple parameters
    Find: Max tools before wrong selection
    """
    print("\n" + "="*60)
    print("AXIS 2: Multiple Tools - Simple Parameters")
    print("="*60)
    
    tool_counts = [1, 5, 10, 20, 30, 50, 75, 100]
    results = {}
    
    for n_tools in tool_counts:
        print(f"\n--- Testing {n_tools} tools ---")
        
        tools = generate_n_simple_tools(n_tools)
        
        # Test: Can it select tool_1 from N tools?
        query = "Use tool_1 with input 'test'"
        
        try:
            calls, final = run_cycle(query, tools, verbose=True)
            
            # Check if correct tool was called
            correct_tool = len(calls) > 0 and "tool_1" in calls[0]
            
            results[n_tools] = {
                "success": correct_tool,
                "called": len(calls) > 0,
                "correct_tool": correct_tool
            }
            
            status = "✓" if correct_tool else "✗"
            print(f"  Result: {status}")
            
        except Exception as e:
            print(f"  Result: ✗ ERROR - {str(e)[:50]}")
            results[n_tools] = {"success": False, "error": str(e)}
            break
    
    # Summary
    print("\n" + "-"*60)
    print("AXIS 2 SUMMARY:")
    for n, result in results.items():
        if result.get("success"):
            print(f"  {n} tools: ✓")
        else:
            print(f"  {n} tools: ✗")
    
    max_success = max([n for n, r in results.items() if r.get("success")], default=0)
    print(f"\nLimit: {max_success} tools")
    print("-"*60)
    
    return results


# ============================================================
# AXIS 3: SEQUENTIAL CALLS - SIMPLE TOOLS
# ============================================================

def test_axis3_sequential_calls():
    """
    Test: Sequential tool calls in single turn
    Find: Max sequential calls before failure
    """
    print("\n" + "="*60)
    print("AXIS 3: Sequential Calls - Simple Tools")
    print("="*60)
    
    call_counts = [1, 2, 3, 5, 7, 10, 15, 20]
    results = {}
    
    # Create enough tools for testing
    tools = generate_n_simple_tools(20)
    
    for n_calls in call_counts:
        print(f"\n--- Testing {n_calls} sequential calls ---")
        
        # Generate query asking for N tool calls
        tool_names = [f"tool_{i+1}" for i in range(n_calls)]
        query = f"Call these tools in order: {', '.join(tool_names)}"
        
        try:
            calls, final = run_cycle(query, tools, verbose=True)
            
            # Check how many calls were made
            actual_calls = len(calls)
            
            # Check if right tools were called
            correct_tools = sum(1 for i in range(min(n_calls, actual_calls)) 
                              if f"tool_{i+1}" in calls[i]) if calls else 0
            
            results[n_calls] = {
                "success": actual_calls >= n_calls,
                "actual_calls": actual_calls,
                "correct_tools": correct_tools
            }
            
            status = "✓" if actual_calls >= n_calls else "✗"
            print(f"  Result: {status} - Got {actual_calls}/{n_calls} calls, {correct_tools} correct")
            
        except Exception as e:
            print(f"  Result: ✗ ERROR - {str(e)[:50]}")
            results[n_calls] = {"success": False, "error": str(e)}
            break
    
    # Summary
    print("\n" + "-"*60)
    print("AXIS 3 SUMMARY:")
    for n, result in results.items():
        if result.get("success"):
            actual = result.get("actual_calls", 0)
            correct = result.get("correct_tools", 0)
            print(f"  {n} calls: ✓ (got {actual}, {correct} correct)")
        else:
            print(f"  {n} calls: ✗")
    
    max_success = max([n for n, r in results.items() if r.get("success")], default=0)
    print(f"\nLimit: {max_success} sequential calls")
    print("-"*60)
    
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    """Run all three axis tests"""
    print("="*60)
    print("TOOL CALLING LIMITS - THREE-AXIS TESTING")
    print(f"Server: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print("="*60)
    
    try:
        axis1_results = test_axis1_parameter_scaling()
        axis2_results = test_axis2_multiple_tools()
        axis3_results = test_axis3_sequential_calls()
        
        # Final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY - MODEL LIMITS")
        print("="*60)
        
        axis1_limit = max([n for n, r in axis1_results.items() if r.get("success")], default=0)
        axis2_limit = max([n for n, r in axis2_results.items() if r.get("success")], default=0)
        axis3_limit = max([n for n, r in axis3_results.items() if r.get("success")], default=0)
        
        print(f"Axis 1 (Parameters):      {axis1_limit} max parameters")
        print(f"Axis 2 (Multiple Tools):  {axis2_limit} max tools")
        print(f"Axis 3 (Sequential):      {axis3_limit} max sequential calls")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
