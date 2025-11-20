"""
MINIMAL MCP EXAMPLE - Fixed for Directory Operations
"""
import subprocess
import json
import requests
import re

# Config
LLM_URL = "http://localhost:8080/v1"
# MODEL = "LFM2-1.2B-Tool-Q4_K_M-cuda"
MODEL = "LFM2-8B-A1B-BF16-cuda"
# MODEL = "LFM2-8B-A1B-UD-Q3_K_XL-cuda"
# MODEL = "LFM2-8B-A1B-UD-Q3_K_XL-cpu"
MCP_CMD = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

# Start MCP server
proc = subprocess.Popen(MCP_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

# Initialize MCP
proc.stdin.write(json.dumps({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}})+"\n")
proc.stdin.flush()
proc.stdout.readline()

# Get tools
proc.stdin.write(json.dumps({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}})+"\n")
proc.stdin.flush()
tools_response = json.loads(proc.stdout.readline())
mcp_tools = tools_response["result"]["tools"]

# Convert to OpenAI format
openai_tools = [{"type":"function","function":{"name":t["name"],"description":t.get("description",""),"parameters":t.get("inputSchema",{})}} for t in mcp_tools]

print(f"MCP Tools: {[t['name'] for t in mcp_tools]}\n")

# TEST 1: List directory
print("="*60)
print("TEST 1: List directory contents")
print("="*60)

conv = [{"role":"user","content":"Use list_directory to show files in /tmp"}]

for i in range(10):
    resp = requests.post(f"{LLM_URL}/chat/completions", json={"model":MODEL,"messages":conv,"tools":openai_tools,"temperature":0,"max_tokens":512}).json()
    content = resp["choices"][0]["message"]["content"]
    
    if "<|tool_call_start|>" not in content:
        print(f"\nAnswer: {content}\n")
        break
    
    calls_str = content.split("<|tool_call_start|>")[1].split("<|tool_call_end|>")[0].strip("[]")
    tool_calls = re.findall(r'\w+\([^)]*\)', calls_str)
    
    print(f"Iter {i+1}: {tool_calls}")
    conv.append({"role":"assistant","content":content})
    
    for call in tool_calls:
        name = call.split("(")[0]
        args = {}
        for match in re.findall(r'(\w+)=(["\'])([^"\']*)\2', call):
            args[match[0]] = match[2]
        
        proc.stdin.write(json.dumps({"jsonrpc":"2.0","id":i+10,"method":"tools/call","params":{"name":name,"arguments":args}})+"\n")
        proc.stdin.flush()
        result = json.loads(proc.stdout.readline())["result"]
        
        print(f"  → {name}({args})")
        print(f"  ← {str(result)[:150]}...")
        conv.append({"role":"tool","content":json.dumps(result)})

# TEST 2: Create and read a file
print("\n" + "="*60)
print("TEST 2: Create file and read it back")
print("="*60)

conv = [{"role":"user","content":"Use write_file to create /tmp/test_mcp.txt with content 'Hello from MCP!', then use read_text_file to read it back"}]

for i in range(10):
    resp = requests.post(f"{LLM_URL}/chat/completions", json={"model":MODEL,"messages":conv,"tools":openai_tools,"temperature":0,"max_tokens":512}).json()
    content = resp["choices"][0]["message"]["content"]
    
    if "<|tool_call_start|>" not in content:
        print(f"\nAnswer: {content}\n")
        break
    
    calls_str = content.split("<|tool_call_start|>")[1].split("<|tool_call_end|>")[0].strip("[]")
    tool_calls = re.findall(r'\w+\([^)]*\)', calls_str)
    
    print(f"Iter {i+1}: {tool_calls}")
    conv.append({"role":"assistant","content":content})
    
    for call in tool_calls:
        name = call.split("(")[0]
        args = {}
        for match in re.findall(r'(\w+)=(["\'])([^"\']*)\2', call):
            args[match[0]] = match[2]
        
        proc.stdin.write(json.dumps({"jsonrpc":"2.0","id":i+20,"method":"tools/call","params":{"name":name,"arguments":args}})+"\n")
        proc.stdin.flush()
        result = json.loads(proc.stdout.readline())["result"]
        
        print(f"  → {name}({list(args.keys())})")
        print(f"  ← {str(result)[:150]}...")
        conv.append({"role":"tool","content":json.dumps(result)})

# TEST 3: Chain operations
print("\n" + "="*60)
print("TEST 3: Chain - List, Write, Read, Get Info")
print("="*60)

conv = [{"role":"user","content":"First list_directory /tmp, then write_file /tmp/chain_test.txt with 'Chain test data', then read_text_file it, then get_file_info on it"}]

for i in range(10):
    resp = requests.post(f"{LLM_URL}/chat/completions", json={"model":MODEL,"messages":conv,"tools":openai_tools,"temperature":0,"max_tokens":512}).json()
    content = resp["choices"][0]["message"]["content"]
    
    if "<|tool_call_start|>" not in content:
        print(f"\nAnswer: {content}\n")
        break
    
    calls_str = content.split("<|tool_call_start|>")[1].split("<|tool_call_end|>")[0].strip("[]")
    tool_calls = re.findall(r'\w+\([^)]*\)', calls_str)
    
    print(f"Iter {i+1}: {tool_calls}")
    conv.append({"role":"assistant","content":content})
    
    for call in tool_calls:
        name = call.split("(")[0]
        args = {}
        for match in re.findall(r'(\w+)=(["\'])([^"\']*)\2', call):
            args[match[0]] = match[2]
        
        proc.stdin.write(json.dumps({"jsonrpc":"2.0","id":i+30,"method":"tools/call","params":{"name":name,"arguments":args}})+"\n")
        proc.stdin.flush()
        result = json.loads(proc.stdout.readline())["result"]
        
        print(f"  → {name}")
        print(f"  ← {str(result)[:100]}...")
        conv.append({"role":"tool","content":json.dumps(result)})

print("\n" + "="*60)
print("ALL TESTS COMPLETE")
print("="*60)

# Cleanup
proc.terminate()