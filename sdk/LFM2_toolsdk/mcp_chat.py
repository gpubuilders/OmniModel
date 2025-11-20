"""
INTERACTIVE MCP CHAT
Real-time chat interface with MCP filesystem tools
Based on working minimal example
"""
import subprocess
import json
import requests
import re
import sys
from datetime import datetime
from typing import List, Dict, Set

# Config
LLM_URL = "http://localhost:8080/v1"
MODEL = "LFM2-8B-A1B-UD-Q3_K_XL-cpu"  # Change to your model
MCP_CMD = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

class MCPChat:
    """Interactive chat with MCP tools"""
    
    def __init__(self):
        print("Starting MCP Filesystem Server...")
        
        # Start MCP server
        self.proc = subprocess.Popen(
            MCP_CMD, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            text=True, 
            bufsize=1
        )
        
        # Initialize MCP
        self.proc.stdin.write(json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "interactive-chat", "version": "1.0"}
            }
        }) + "\n")
        self.proc.stdin.flush()
        self.proc.stdout.readline()
        
        # Get tools
        self.proc.stdin.write(json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }) + "\n")
        self.proc.stdin.flush()
        tools_response = json.loads(self.proc.stdout.readline())
        self.mcp_tools = tools_response["result"]["tools"]
        
        # Convert to OpenAI format
        self.openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema", {})
                }
            }
            for t in self.mcp_tools
        ]
        
        # Chat state
        self.conversation = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant with access to filesystem tools. "
                    "You can read, write, list, and manage files in /tmp. "
                    "Always use the appropriate tools to help the user. "
                    "Be conversational and helpful."
                )
            }
        ]
        self.tool_id = 100
        self.called_tools: Set[str] = set()  # Anti-loop
        
        print(f"✓ MCP Server ready")
        print(f"✓ Available tools: {len(self.mcp_tools)}")
        print(f"  {', '.join([t['name'] for t in self.mcp_tools[:5]])}...")
    
    def call_tool(self, name: str, args: Dict) -> Dict:
        """Execute tool via MCP"""
        self.tool_id += 1
        self.proc.stdin.write(json.dumps({
            "jsonrpc": "2.0",
            "id": self.tool_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": args}
        }) + "\n")
        self.proc.stdin.flush()
        result = json.loads(self.proc.stdout.readline())["result"]
        return result
    
    def process_message(self, user_input: str, verbose: bool = False) -> str:
        """Process user message and return response"""
        
        # Add user message
        self.conversation.append({"role": "user", "content": user_input})
        
        # Reset anti-loop for new query
        self.called_tools.clear()
        
        # Process with tools (max 10 iterations to prevent infinite loops)
        for iteration in range(10):
            if verbose:
                print(f"\n  [Iteration {iteration + 1}]")
            
            # Call LLM
            try:
                response = requests.post(
                    f"{LLM_URL}/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": self.conversation,
                        "tools": self.openai_tools,
                        "temperature": 0.7,
                        "max_tokens": 512
                    },
                    timeout=30
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                return f"Error calling LLM: {e}"
            
            # Check if model wants to call tools
            if "<|tool_call_start|>" not in content:
                # Final answer - add to conversation and return
                self.conversation.append({"role": "assistant", "content": content})
                return content
            
            # Extract tool calls
            calls_str = content.split("<|tool_call_start|>")[1].split("<|tool_call_end|>")[0].strip("[]")
            tool_calls = re.findall(r'\w+\([^)]*\)', calls_str)
            
            if verbose:
                print(f"  Tool calls: {tool_calls}")
            
            self.conversation.append({"role": "assistant", "content": content})
            
            # Execute tools
            for call in tool_calls:
                # Parse tool call
                name = call.split("(")[0]
                args = {}
                for match in re.findall(r'(\w+)=(["\'])([^"\']*)\2', call):
                    args[match[0]] = match[2]
                
                # Anti-loop: skip if already called
                call_signature = f"{name}({json.dumps(args, sort_keys=True)})"
                if call_signature in self.called_tools:
                    if verbose:
                        print(f"  ⚠️  Skipping duplicate: {call}")
                    self.conversation.append({
                        "role": "tool",
                        "content": json.dumps({"note": "Already called this tool"})
                    })
                    continue
                
                self.called_tools.add(call_signature)
                
                # Execute tool
                try:
                    result = self.call_tool(name, args)
                    if verbose:
                        result_str = json.dumps(result)
                        print(f"  → {name}({args})")
                        print(f"  ← {result_str[:100]}...")
                    self.conversation.append({
                        "role": "tool",
                        "content": json.dumps(result)
                    })
                except Exception as e:
                    if verbose:
                        print(f"  ✗ Tool error: {e}")
                    self.conversation.append({
                        "role": "tool",
                        "content": json.dumps({"error": str(e)})
                    })
        
        # Max iterations reached
        return "I apologize, I got stuck in a loop. Let me try rephrasing that..."
    
    def run(self):
        """Main interactive loop"""
        print("\n" + "="*60)
        print("MCP INTERACTIVE CHAT")
        print("="*60)
        print("Chat with your AI assistant. It has access to filesystem tools.")
        print("Working directory: /tmp")
        print("\nCommands:")
        print("  /help    - Show available tools")
        print("  /clear   - Clear conversation history")
        print("  /verbose - Toggle verbose mode")
        print("  /quit    - Exit chat")
        print("="*60 + "\n")
        
        verbose = False
        
        try:
            while True:
                # Get user input
                try:
                    user_input = input("\n\033[1;34mYou:\033[0m ").strip()
                except EOFError:
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    if user_input == "/quit":
                        print("\nGoodbye!")
                        break
                    
                    elif user_input == "/help":
                        print("\n\033[1;32mAvailable Tools:\033[0m")
                        for tool in self.mcp_tools:
                            print(f"  • {tool['name']}: {tool.get('description', 'No description')[:60]}...")
                        continue
                    
                    elif user_input == "/clear":
                        self.conversation = [self.conversation[0]]  # Keep system message
                        print("\n\033[1;33m[Conversation cleared]\033[0m")
                        continue
                    
                    elif user_input == "/verbose":
                        verbose = not verbose
                        print(f"\n\033[1;33m[Verbose mode: {'ON' if verbose else 'OFF'}]\033[0m")
                        continue
                    
                    else:
                        print(f"\n\033[1;31mUnknown command: {user_input}\033[0m")
                        continue
                
                # Process message
                print("\n\033[1;32mAssistant:\033[0m ", end="", flush=True)
                
                response = self.process_message(user_input, verbose=verbose)
                
                print(response)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except:
            self.proc.kill()
        print("\n✓ MCP Server stopped")


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_conversation():
    """Show example conversation flow"""
    chat = MCPChat()
    
    print("\n" + "="*60)
    print("EXAMPLE CONVERSATION")
    print("="*60)
    
    examples = [
        "List all files in /tmp",
        "Create a file called hello.txt with the content 'Hello World'",
        "Read the file you just created",
        "What files are in /tmp now?"
    ]
    
    for query in examples:
        print(f"\n\033[1;34mYou:\033[0m {query}")
        print(f"\033[1;32mAssistant:\033[0m ", end="", flush=True)
        response = chat.process_message(query, verbose=True)
        print(response)
    
    chat.cleanup()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "example":
        # Run example conversation
        example_conversation()
    else:
        # Run interactive chat
        chat = MCPChat()
        chat.run()
