"""
INTERACTIVE MCP CHAT - Enhanced Tool Transparency
Shows Every Step: Tool Call → Raw Result → LLM Interpretation
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
# MODEL = "LFM2-8B-A1B-UD-Q3_K_XL-cpu"  # Change to your model

MODEL = "LFM2-8B-A1B-BF16-cuda"
MCP_CMD = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

class MCPChat:
    """Interactive chat with MCP tools - Now with full transparency!"""
    
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
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
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
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}
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
        
        # In MCPChat.__init__ after getting tools
        tools_list = ", ".join([f"{t['name']}" for t in self.mcp_tools])
        tools_examples = "\n".join([
            f"- {t['name']}: {t.get('description', 'No description')}" 
            for t in self.mcp_tools
        ])

        self.conversation = [{
            "role": "system",
            "content": (
                f"You are a precise AI assistant with filesystem tools: {tools_list}.\n\n"
                "CRITICAL RULES: Always assume the user wants to call a tool, and call it!\n"
                "1. For ANY file/directory request, you MUST call a tool\n"
                "2. NEVER guess or make up information\n"
                "3. Provide conversational answers based on tool results\n\n"
                "EXAMPLE FLOW:\n"
                "User: 'List files in /tmp'\n"
                "You: <|tool_call_start|>[list_directory(path=\"/tmp\")] <|tool_call_end|> \n"
                f"Tool: {{\"contents\": [...]}}\n"
                "You: 'I found X files...'"
            )
        }]
        self.tool_id = 100
        self.called_tools: Set[str] = set()
        self.debug_mode = False
        
        print(f"✓ MCP Server ready")
        print(f"✓ {len(self.mcp_tools)} tools available")
        print(f"  {', '.join([t['name'] for t in self.mcp_tools])}")
    
    def call_tool(self, name: str, args: Dict) -> Dict:
        """Execute tool via MCP"""
        self.tool_id += 1
        self.proc.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": self.tool_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": args}
        }) + "\n")
        self.proc.stdin.flush()
        result = json.loads(self.proc.stdout.readline())["result"]
        return result
    
    def process_message(self, user_input: str) -> str:
        """Process user message with full transparency mode"""
        
        # Add user message
        self.conversation.append({"role": "user", "content": user_input})
        self.called_tools.clear()
        
        print("\n" + "─" * 60)
        print(f"\033[1;34m📝 USER REQUEST:\033[0m {user_input}")
        print("─" * 60)
        
        # Process with tools (max 10 iterations)
        for iteration in range(10):
            if self.debug_mode:
                print(f"\n\033[1;36m[Debug] Iteration {iteration + 1} - Calling LLM...\033[0m")
            
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
                return f"❌ Error calling LLM: {e}"
            
            # Check if model wants to call tools
            if "<|tool_call_start|>" not in content:
                # FINAL ANSWER - No more tool calls
                print(f"\n\033[1;32m✓ FINAL RESPONSE:\033[0m")
                print("─" * 60)
                print(content)
                print("─" * 60)
                self.conversation.append({"role": "assistant", "content": content})
                return content
            
            # ─────────────────────────────────────────────────────────────
            # TOOL CALL DETECTED
            # ─────────────────────────────────────────────────────────────
            print(f"\n\033[1;33m🔧 TOOL CALL DETECTED [Iteration {iteration + 1}]\033[0m")
            
            # Parse tool calls
            calls_str = content.split("<|tool_call_start|>")[1].split(" <|tool_call_end|> ")[0].strip("[]")
            tool_calls = re.findall(r'\w+\([^)]*\)', calls_str)
            
            print("\033[1;33mLLM wants to call:\033[0m")
            for call in tool_calls:
                print(f"  → {call}")
            
            # Add assistant's tool call request to conversation
            self.conversation.append({"role": "assistant", "content": content})
            
            # Execute each tool call
            for call in tool_calls:
                # Parse tool name and arguments
                name = call.split("(")[0]
                args = {}
                for match in re.findall(r'(\w+)=(["\'])([^"\']*)\2', call):
                    args[match[0]] = match[2]
                
                # Anti-loop check
                call_signature = f"{name}({json.dumps(args, sort_keys=True)})"
                if call_signature in self.called_tools:
                    print(f"\n⚠️  \033[1;31mANTI-LOOP:\033[0m Skipping duplicate call")
                    tool_result = {"note": "Already called this tool"}
                else:
                    self.called_tools.add(call_signature)
                    
                    # Execute tool
                    print(f"\n\033[1;35m📡 EXECUTING TOOL:\033[0m {name}")
                    print(f"   Arguments: {args}")
                    try:
                        tool_result = self.call_tool(name, args)
                    except Exception as e:
                        tool_result = {"error": str(e)}
                        print(f"   ❌ Error: {e}")
                
                # Show raw tool result
                print(f"\033[1;35m📥 RAW TOOL RESULT:\033[0m")
                print(f"   {json.dumps(tool_result, indent=4)}")
                
                # Add tool result to conversation (for LLM to interpret)
                self.conversation.append({
                    "role": "tool",
                    "content": json.dumps(tool_result)
                })
        
        # Max iterations reached
        return "❌ I apologize, I got stuck in a loop. Let me try rephrasing that..."
    
    def run(self):
        """Main interactive loop"""
        print("\n" + "="*60)
        print("MCP INTERACTIVE CHAT - Full Transparency Mode")
        print("="*60)
        print("Working directory: /tmp")
        print("\nCommands:")
        print("  /help    - Show available tools")
        print("  /clear   - Clear conversation history")
        print("  /debug   - Toggle debug mode")
        print("  /verbose - Toggle verbose mode (legacy)")
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
                        print("\n👋 Goodbye!")
                        break
                    
                    elif user_input == "/help":
                        print("\n\033[1;32m📋 Available Tools:\033[0m")
                        for tool in self.mcp_tools:
                            print(f"  ✦ {tool['name']}")
                            print(f"    {tool.get('description', 'No description')[:60]}...")
                        continue
                    
                    elif user_input == "/clear":
                        self.conversation = [self.conversation[0]]
                        print("\n\033[1;33m🗑️  Conversation cleared\033[0m")
                        continue
                    
                    elif user_input == "/debug":
                        self.debug_mode = not self.debug_mode
                        print(f"\n\033[1;33m[Debug mode: {'ON' if self.debug_mode else 'OFF'}]\033[0m")
                        continue
                    
                    elif user_input == "/verbose":
                        verbose = not verbose
                        print(f"\n\033[1;33m[Verbose mode: {'ON' if verbose else 'OFF'}]\033[0m")
                        continue
                    
                    else:
                        print(f"\n\033[1;31m❌ Unknown command: {user_input}\033[0m")
                        continue
                
                # Process message
                response = self.process_message(user_input)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Exiting...")
        
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

def demonstrate_tool_flow():
    """Show a complete example with all steps visible"""
    print("\n" + "═"*70)
    print("DEMONSTRATION: Complete Tool Call Flow")
    print("═"*70)
    print("This shows every step: User → LLM → Tool → Result → LLM → Answer")
    
    chat = MCPChat()
    chat.debug_mode = True  # Enable debug for demonstration
    
    # Simulate a user's request
    request = "Check if /tmp/test.txt exists, and if not, create it with 'Hello MCP!'"
    
    print(f"\n🎯 Example Request: '{request}'")
    print("═"*70)
    
    # Process and show full flow
    response = chat.process_message(request)
    
    chat.cleanup()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run demonstration
        demonstrate_tool_flow()
    else:
        # Run interactive chat
        chat = MCPChat()
        chat.run()