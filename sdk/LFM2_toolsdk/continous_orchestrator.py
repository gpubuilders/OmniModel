"""
Production Tool Chain Orchestrator
Runs continuously with discovered limits in mind
"""
import requests
import json
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum
import re

# ============================================================
# CONFIGURATION
# ============================================================
BASE_URL = "http://localhost:8080/v1"
MODEL_NAME = "LFM2-1.2B-Tool-Q4_K_M-cuda"

# Limits discovered from testing (adjust based on your test results)
MAX_CHAIN_DEPTH = 8  # Maximum tool chain before reset
MAX_HISTORY_MESSAGES = 20  # Maximum messages before summarization
RECALL_SAFE_DEPTH = 5  # Steps back that model can reliably recall


# ============================================================
# CHAIN STATE MANAGER
# ============================================================
class ChainState:
    """Manages conversation state with limit awareness"""
    
    def __init__(self, max_depth: int = MAX_CHAIN_DEPTH, 
                 max_history: int = MAX_HISTORY_MESSAGES):
        self.conversation: List[Dict] = []
        self.chain_depth = 0
        self.max_depth = max_depth
        self.max_history = max_history
        self.tool_results_cache: Dict[str, str] = {}
        self.last_n_results: List[Dict] = []  # For recall testing
        
    def add_message(self, role: str, content: str):
        """Add message and check if limits exceeded"""
        self.conversation.append({"role": role, "content": content})
        
        if role == "assistant" and "<|tool_call_start|>" in content:
            self.chain_depth += 1
            
        # Check if we need to compress history
        if len(self.conversation) > self.max_history:
            self._compress_history()
            
    def cache_tool_result(self, tool_call: str, result: str):
        """Cache recent tool results for quick recall"""
        self.tool_results_cache[tool_call] = result
        self.last_n_results.append({
            "depth": self.chain_depth,
            "call": tool_call,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last RECALL_SAFE_DEPTH results
        if len(self.last_n_results) > RECALL_SAFE_DEPTH:
            self.last_n_results.pop(0)
    
    def _compress_history(self):
        """Compress old conversation history to stay within limits"""
        # Keep first message (system/instructions)
        # Keep last N messages
        # Summarize middle section
        keep_recent = 10
        
        if len(self.conversation) <= keep_recent + 1:
            return
            
        summary = self._create_summary(
            self.conversation[1:-keep_recent]
        )
        
        self.conversation = [
            self.conversation[0],  # First message
            {"role": "system", "content": f"Previous context summary: {summary}"},
            *self.conversation[-keep_recent:]  # Recent messages
        ]
        
    def _create_summary(self, messages: List[Dict]) -> str:
        """Create summary of conversation segment"""
        tool_calls = []
        for msg in messages:
            if msg["role"] == "assistant" and "<|tool_call_start|>" in msg["content"]:
                calls = extract_tool_calls(msg["content"])
                tool_calls.extend(calls)
        
        return f"Executed {len(tool_calls)} tool calls: {', '.join(tool_calls[:5])}"
    
    def should_reset_chain(self) -> bool:
        """Check if chain should be reset"""
        return self.chain_depth >= self.max_depth
    
    def reset_chain(self):
        """Reset chain depth counter"""
        self.chain_depth = 0
        # Optionally keep cached results
        

# ============================================================
# CORE API FUNCTIONS
# ============================================================
def call_api(messages: List[Dict], tools: Optional[List[Dict]] = None, 
             temperature: float = 0) -> str:
    """API call with error handling"""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
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
        response.raise_for_status()
        data = response.json()
        
        if "error" in data:
            raise Exception(f"API Error: {data['error']['message']}")
            
        return data["choices"][0]["message"]["content"]
    
    except requests.exceptions.Timeout:
        raise Exception("API timeout")
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")


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


# ============================================================
# TOOL EXECUTION ENGINE
# ============================================================
class ToolExecutor:
    """Execute tools with realistic mock data"""
    
    def __init__(self, data_source: Dict):
        self.data = data_source
        
    def execute(self, tool_call: str) -> str:
        """Execute a tool call and return result"""
        tool_name = tool_call.split("(")[0]
        param_value = self._extract_param(tool_call)
        
        # Route to appropriate handler
        handlers = {
            "lookup_user": lambda p: self.data["users"].get(p, {"error": "User not found"}),
            "get_user_orders": lambda p: self.data["orders"].get(p, []),
            "get_order_details": lambda p: self.data["order_details"].get(p, {"error": "Order not found"}),
            "check_inventory": lambda p: self.data["inventory"].get(p, {"error": "Product not found"}),
            "get_supplier_info": lambda p: self.data["suppliers"].get(p, {"error": "Supplier not found"}),
            "get_contact_details": lambda p: self.data["contacts"].get(p, {"error": "Contact not found"}),
            "get_territory_info": lambda p: self.data["territories"].get(p, {"error": "Territory not found"}),
            "get_manager_info": lambda p: self.data["managers"].get(p, {"error": "Manager not found"}),
        }
        
        handler = handlers.get(tool_name)
        if handler:
            result = handler(param_value)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
            
        return json.dumps(result)
    
    def _extract_param(self, tool_call: str) -> Optional[str]:
        """Extract parameter value from tool call"""
        if '="' in tool_call or "='" in tool_call:
            parts = tool_call.split('="') if '="' in tool_call else tool_call.split("='")
            if len(parts) > 1:
                return parts[1].split('"')[0] if '="' in tool_call else parts[1].split("'")[0]
        return None


# ============================================================
# ORCHESTRATOR
# ============================================================
class ToolChainOrchestrator:
    """Main orchestrator for continuous operation"""
    
    def __init__(self, tools: List[Dict], data_source: Dict):
        self.tools = tools
        self.executor = ToolExecutor(data_source)
        self.state = ChainState()
        
    def process_query(self, query: str) -> str:
        """Process a single query with tool chaining"""
        print(f"\n{'='*60}")
        print(f"Processing: {query}")
        print(f"Current chain depth: {self.state.chain_depth}/{self.state.max_depth}")
        print(f"{'='*60}")
        
        # Check if we need to reset chain
        if self.state.should_reset_chain():
            print("⚠️  Chain depth limit reached - resetting chain")
            self.state.reset_chain()
            # Add a summary message
            self.state.add_message(
                "system",
                "Chain reset. Previous tool results are still available."
            )
        
        # Add user query
        self.state.add_message("user", query)
        
        # Process with tools (allow multiple iterations)
        max_iterations = 5  # Prevent infinite loops
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Get model response
            response = call_api(self.state.conversation, self.tools)
            
            # Check for tool calls
            tool_calls = extract_tool_calls(response)
            
            if not tool_calls:
                # No more tool calls - we have final answer
                print("✓ Final answer generated")
                self.state.add_message("assistant", response)
                return response
            
            # Execute tools
            print(f"Tool calls: {tool_calls}")
            self.state.add_message("assistant", response)
            
            for tool_call in tool_calls:
                result = self.executor.execute(tool_call)
                print(f"  → {tool_call}: {result[:80]}...")
                
                self.state.add_message("tool", result)
                self.state.cache_tool_result(tool_call, result)
        
        # If we exhausted iterations, return last response
        print("⚠️  Max iterations reached")
        return response
    
    def run_continuously(self, query_stream):
        """Process queries continuously from a stream"""
        for query in query_stream:
            try:
                result = self.process_query(query)
                yield {
                    "query": query,
                    "result": result,
                    "chain_depth": self.state.chain_depth,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                print(f"✗ Error processing query: {str(e)}")
                yield {
                    "query": query,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }


# ============================================================
# USE CASE IMPLEMENTATIONS
# ============================================================

class CustomerSupportAgent:
    """Customer support use case"""
    
    def __init__(self, orchestrator: ToolChainOrchestrator):
        self.orchestrator = orchestrator
    
    def handle_order_inquiry(self, email: str, question: str) -> str:
        """Handle customer order questions"""
        query = f"Customer {email} asks: {question}. Please look up their account and help answer."
        return self.orchestrator.process_query(query)


class InventoryMonitor:
    """Inventory monitoring use case"""
    
    def __init__(self, orchestrator: ToolChainOrchestrator):
        self.orchestrator = orchestrator
        
    def check_low_stock_suppliers(self) -> List[Dict]:
        """Check for low stock items and get supplier contact info"""
        query = "Check all products in inventory. For any below reorder point, get supplier contact details."
        result = self.orchestrator.process_query(query)
        return result


# ============================================================
# EXAMPLE USAGE
# ============================================================
def main():
    """Example of continuous operation"""
    
    # Import realistic data from original test
    from __main__ import REALISTIC_DATA, CHAIN_TOOLS
    
    # Create orchestrator
    orchestrator = ToolChainOrchestrator(CHAIN_TOOLS, REALISTIC_DATA)
    
    # Example 1: Customer Support Agent
    print("\n" + "="*60)
    print("USE CASE 1: Customer Support")
    print("="*60)
    
    support_agent = CustomerSupportAgent(orchestrator)
    
    customer_queries = [
        ("sarah.chen@techcorp.com", "What's the status of my recent order?"),
        ("sarah.chen@techcorp.com", "What items did I order?"),
        ("mike.rodriguez@startup.io", "When will my order arrive?"),
    ]
    
    for email, question in customer_queries:
        result = support_agent.handle_order_inquiry(email, question)
        print(f"\n✓ Response: {result[:200]}...")
        time.sleep(1)  # Rate limiting
    
    # Example 2: Continuous Query Stream
    print("\n" + "="*60)
    print("USE CASE 2: Continuous Query Processing")
    print("="*60)
    
    query_stream = [
        "Who is the supplier for wireless keyboards?",
        "What's the stock level of USB-C cables?",
        "Get contact info for the Logitech supplier",
    ]
    
    for result in orchestrator.run_continuously(query_stream):
        print(f"\n✓ Processed: {result['query']}")
        print(f"  Chain depth: {result['chain_depth']}")
        print(f"  Result: {result.get('result', result.get('error'))[:150]}...")
        time.sleep(1)


if __name__ == "__main__":
    main()