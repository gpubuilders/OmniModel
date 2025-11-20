"""
CHAIN DEPTH DISCOVERY - Real MCP Operations
Discovers actual chain depth limits with real tool calls
"""
import subprocess
import json
import requests
import re
import time


# this is broken

LLM_URL = "http://localhost:8080/v1"
# MODEL = "LFM2-1.2B-Tool-Q4_K_M-cuda"
# MODEL = "LFM2-8B-A1B-BF16-cuda"
# MODEL = "LFM2-8B-A1B-UD-Q3_K_XL-cuda"
MODEL = "LFM2-8B-A1B-UD-Q3_K_XL-cpu"
MCP_CMD = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

# Start MCP
proc = subprocess.Popen(MCP_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

# Initialize
proc.stdin.write(json.dumps({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}})+"\n")
proc.stdin.flush()
proc.stdout.readline()

# Get tools
proc.stdin.write(json.dumps({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}})+"\n")
proc.stdin.flush()
tools_response = json.loads(proc.stdout.readline())
mcp_tools = tools_response["result"]["tools"]
openai_tools = [{"type":"function","function":{"name":t["name"],"description":t.get("description",""),"parameters":t.get("inputSchema",{})}} for t in mcp_tools]

print("MCP Filesystem Server Connected")
print("="*60)

def execute_tool_via_mcp(name, args, req_id):
    """Execute single tool via MCP"""
    proc.stdin.write(json.dumps({"jsonrpc":"2.0","id":req_id,"method":"tools/call","params":{"name":name,"arguments":args}})+"\n")
    proc.stdin.flush()
    result = json.loads(proc.stdout.readline())["result"]
    return result

def test_chain_depth(max_depth=15):
    """Test progressively deeper chains"""
    
    print("\nTEST: CHAIN DEPTH DISCOVERY")
    print("="*60)
    
    results = []
    
    for depth in range(1, max_depth + 1):
        print(f"\n--- Testing Chain Depth: {depth} ---")
        
        # Create test files for this depth
        for i in range(depth):
            execute_tool_via_mcp(
                "write_file",
                {"path": f"/tmp/depth_test_{i}.txt", "content": f"File {i} data"},
                1000 + i
            )
        
        # Build query that requires reading all files in sequence
        file_list = ", ".join([f"/tmp/depth_test_{i}.txt" for i in range(depth)])
        query = f"Read these {depth} files in order and tell me their contents: {file_list}"
        
        conv = [{"role": "user", "content": query}]
        
        # Track execution
        tools_called = []
        success = False
        
        try:
            for iteration in range(depth + 5):  # Allow some extra iterations
                resp = requests.post(
                    f"{LLM_URL}/chat/completions",
                    json={"model": MODEL, "messages": conv, "tools": openai_tools, "temperature": 0, "max_tokens": 512}
                ).json()
                
                content = resp["choices"][0]["message"]["content"]
                
                if "<|tool_call_start|>" not in content:
                    # Got final answer
                    success = True
                    print(f"  ✓ Final answer after {len(tools_called)} tool calls")
                    break
                
                # Extract and execute tools
                calls_str = content.split("<|tool_call_start|>")[1].split("<|tool_call_end|>")[0].strip("[]")
                tool_calls = re.findall(r'\w+\([^)]*\)', calls_str)
                
                conv.append({"role": "assistant", "content": content})
                
                for call in tool_calls:
                    name = call.split("(")[0]
                    args = {}
                    for match in re.findall(r'(\w+)=(["\'])([^"\']*)\2', call):
                        args[match[0]] = match[2]
                    
                    result = execute_tool_via_mcp(name, args, 2000 + len(tools_called))
                    tools_called.append(name)
                    
                    conv.append({"role": "tool", "content": json.dumps(result)})
                
                print(f"  Iteration {iteration + 1}: {len(tool_calls)} calls, total: {len(tools_called)}")
        
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
        
        # Record result
        result_data = {
            "depth": depth,
            "success": success,
            "tools_called": len(tools_called),
            "expected_tools": depth
        }
        results.append(result_data)
        
        if success:
            print(f"  ✓ SUCCESS - Called {len(tools_called)} tools for depth {depth}")
        else:
            print(f"  ✗ FAILED - Chain broke at depth {depth}")
            print(f"  Max reliable depth: {depth - 1}")
            break
        
        time.sleep(0.5)  # Rate limit
    
    return results

def test_history_retention():
    """Test how far back model can recall results"""
    
    print("\n\nTEST: HISTORY RETENTION")
    print("="*60)
    
    # Create sequence of files with unique data
    file_data = {}
    for i in range(8):
        data = f"UNIQUE_DATA_{i}_{'X' * (i * 3)}"
        execute_tool_via_mcp(
            "write_file",
            {"path": f"/tmp/history_test_{i}.txt", "content": data},
            3000 + i
        )
        file_data[i] = data
    
    # Read all files
    conv = [{"role": "user", "content": "Read files /tmp/history_test_0.txt through /tmp/history_test_7.txt"}]
    
    print("\nReading 8 files...")
    for iteration in range(20):
        resp = requests.post(
            f"{LLM_URL}/chat/completions",
            json={"model": MODEL, "messages": conv, "tools": openai_tools, "temperature": 0, "max_tokens": 512}
        ).json()
        
        content = resp["choices"][0]["message"]["content"]
        
        if "<|tool_call_start|>" not in content:
            print(f"✓ All files read in {iteration + 1} iterations\n")
            break
        
        calls_str = content.split("<|tool_call_start|>")[1].split("<|tool_call_end|>")[0].strip("[]")
        tool_calls = re.findall(r'\w+\([^)]*\)', calls_str)
        
        conv.append({"role": "assistant", "content": content})
        
        for call in tool_calls:
            name = call.split("(")[0]
            args = {}
            for match in re.findall(r'(\w+)=(["\'])([^"\']*)\2', call):
                args[match[0]] = match[2]
            
            result = execute_tool_via_mcp(name, args, 4000 + iteration)
            conv.append({"role": "tool", "content": json.dumps(result)})
    
    # Now test recall
    retention_results = []
    
    for test_idx in [0, 2, 4, 6, 7]:
        expected_data = file_data[test_idx]
        query = f"What was the content of history_test_{test_idx}.txt that you read earlier?"
        
        conv.append({"role": "user", "content": query})
        
        resp = requests.post(
            f"{LLM_URL}/chat/completions",
            json={"model": MODEL, "messages": conv, "tools": openai_tools, "temperature": 0, "max_tokens": 512}
        ).json()
        
        content = resp["choices"][0]["message"]["content"]
        conv.append({"role": "assistant", "content": content})
        
        # Check if correct data mentioned
        recalled = expected_data in content
        
        print(f"Recall file {test_idx}: {'✓' if recalled else '✗'}")
        print(f"  Expected: {expected_data}")
        print(f"  Got: {content[:80]}...")
        
        retention_results.append({
            "file_index": test_idx,
            "recalled": recalled
        })
    
    return retention_results

# Run tests
try:
    chain_results = test_chain_depth(max_depth=15)
    retention_results = test_history_retention()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    max_depth = max([r["depth"] for r in chain_results if r["success"]], default=0)
    print(f"\nMax Chain Depth: {max_depth} operations")
    
    recalls = sum([1 for r in retention_results if r["recalled"]])
    print(f"History Retention: {recalls}/{len(retention_results)} successful recalls")
    
    print("\nDetailed Results:")
    print("\nChain Depth:")
    for r in chain_results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} Depth {r['depth']}: {r['tools_called']} tools called")
    
    print("\nHistory Retention:")
    for r in retention_results:
        status = "✓" if r["recalled"] else "✗"
        print(f"  {status} File {r['file_index']}")
    
    print("\n" + "="*60)

finally:
    proc.terminate()
