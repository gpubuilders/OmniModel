"""
Tool Chaining Limits - Realistic Data

Tests two limits:
1. Chain Depth: How many tools can be chained before model loses track?
2. History Retention: How far back can model recall tool results?

Uses realistic mock data to avoid pattern detection.
"""

import requests
import json
from typing import List, Dict, Optional, Tuple
import re
import random


BASE_URL = "http://localhost:8080/v1"
MODEL_NAME = "LFM2-1.2B-Tool-Q4_K_M-cuda"
# MODEL_NAME = "LFM2-8B-A1B-BF16-cuda"

# ============================================================
# REALISTIC MOCK DATA
# ============================================================

REALISTIC_DATA = {
    # Step 1: Find user by email
    "users": {
        "sarah.chen@techcorp.com": {
            "user_id": "usr_8k2m9p4",
            "name": "Sarah Chen",
            "account_type": "premium",
            "customer_since": "2021-03-15"
        },
        "mike.rodriguez@startup.io": {
            "user_id": "usr_3n7q1r8",
            "name": "Mike Rodriguez",
            "account_type": "enterprise",
            "customer_since": "2020-11-22"
        }
    },
    
    # Step 2: Get orders for user_id
    "orders": {
        "usr_8k2m9p4": [
            {"order_id": "ord_x9j2k1", "date": "2024-11-10", "total": 249.99, "status": "delivered"},
            {"order_id": "ord_p4m8n3", "date": "2024-10-22", "total": 89.50, "status": "delivered"},
        ],
        "usr_3n7q1r8": [
            {"order_id": "ord_q7w5e2", "date": "2024-11-15", "total": 1299.00, "status": "shipped"},
        ]
    },
    
    # Step 3: Get order details
    "order_details": {
        "ord_x9j2k1": {
            "items": [
                {"product_id": "prod_wireless_kb", "name": "Wireless Keyboard", "quantity": 2, "price": 79.99},
                {"product_id": "prod_mouse_mx3", "name": "MX Master 3 Mouse", "quantity": 1, "price": 99.99}
            ],
            "shipping_address": "742 Evergreen Terrace",
            "warehouse_id": "wh_seattle_01"
        },
        "ord_p4m8n3": {
            "items": [
                {"product_id": "prod_usbc_cable", "name": "USB-C Cable 6ft", "quantity": 3, "price": 29.99}
            ],
            "shipping_address": "742 Evergreen Terrace",
            "warehouse_id": "wh_portland_02"
        },
        "ord_q7w5e2": {
            "items": [
                {"product_id": "prod_laptop_pro", "name": "MacBook Pro 16", "quantity": 1, "price": 1299.00}
            ],
            "shipping_address": "123 Market Street",
            "warehouse_id": "wh_austin_01"
        }
    },
    
    # Step 4: Get product inventory
    "inventory": {
        "prod_wireless_kb": {
            "stock": 847,
            "supplier_id": "sup_logitech",
            "reorder_point": 200
        },
        "prod_mouse_mx3": {
            "stock": 423,
            "supplier_id": "sup_logitech",
            "reorder_point": 150
        },
        "prod_usbc_cable": {
            "stock": 2891,
            "supplier_id": "sup_anker",
            "reorder_point": 500
        },
        "prod_laptop_pro": {
            "stock": 67,
            "supplier_id": "sup_apple",
            "reorder_point": 20
        }
    },
    
    # Step 5: Get supplier info
    "suppliers": {
        "sup_logitech": {
            "name": "Logitech International",
            "contact_id": "cnt_jl892m",
            "lead_time_days": 7,
            "rating": 4.8
        },
        "sup_anker": {
            "name": "Anker Innovations",
            "contact_id": "cnt_pk234n",
            "lead_time_days": 5,
            "rating": 4.9
        },
        "sup_apple": {
            "name": "Apple Inc.",
            "contact_id": "cnt_am567p",
            "lead_time_days": 14,
            "rating": 4.7
        }
    },
    
    # Step 6: Get contact details
    "contacts": {
        "cnt_jl892m": {
            "name": "Jennifer Lewis",
            "email": "j.lewis@logitech.com",
            "phone": "+1-555-0142",
            "territory_id": "ter_west_coast"
        },
        "cnt_pk234n": {
            "name": "Peter Kim",
            "email": "p.kim@anker.com",
            "phone": "+1-555-0198",
            "territory_id": "ter_international"
        },
        "cnt_am567p": {
            "name": "Amanda Martinez",
            "email": "a.martinez@apple.com",
            "phone": "+1-555-0176",
            "territory_id": "ter_nationwide"
        }
    },
    
    # Step 7: Get territory info
    "territories": {
        "ter_west_coast": {
            "name": "West Coast Region",
            "manager_id": "mgr_smith_j",
            "coverage": ["CA", "OR", "WA"]
        },
        "ter_international": {
            "name": "International Markets",
            "manager_id": "mgr_patel_r",
            "coverage": ["APAC", "EMEA"]
        },
        "ter_nationwide": {
            "name": "US Nationwide",
            "manager_id": "mgr_johnson_m",
            "coverage": ["All US States"]
        }
    },
    
    # Step 8: Get manager info
    "managers": {
        "mgr_smith_j": {
            "name": "James Smith",
            "title": "Regional Sales Director",
            "department_id": "dept_sales_west"
        },
        "mgr_patel_r": {
            "name": "Raj Patel",
            "title": "International Sales VP",
            "department_id": "dept_sales_intl"
        },
        "mgr_johnson_m": {
            "name": "Michelle Johnson",
            "title": "National Accounts Director",
            "department_id": "dept_sales_national"
        }
    }
}


# ============================================================
# CORE FUNCTIONS
# ============================================================

def call_api(messages: List[Dict], tools: Optional[List[Dict]] = None) -> str:
    """Raw API call"""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        # "top_p" : 0.95,
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
    """Execute tool with realistic mock data"""
    
    # Parse tool name
    tool_name = tool_call.split("(")[0]
    
    # Extract parameter value (simple extraction)
    if '="' in tool_call or "='" in tool_call:
        # Find the value between quotes
        parts = tool_call.split('="') if '="' in tool_call else tool_call.split("='")
        if len(parts) > 1:
            param_value = parts[1].split('"')[0] if '="' in tool_call else parts[1].split("'")[0]
        else:
            param_value = None
    else:
        param_value = None
    
    # Route to appropriate data source
    if tool_name == "lookup_user":
        result = REALISTIC_DATA["users"].get(param_value, {"error": "User not found"})
        
    elif tool_name == "get_user_orders":
        result = REALISTIC_DATA["orders"].get(param_value, [])
        
    elif tool_name == "get_order_details":
        result = REALISTIC_DATA["order_details"].get(param_value, {"error": "Order not found"})
        
    elif tool_name == "check_inventory":
        result = REALISTIC_DATA["inventory"].get(param_value, {"error": "Product not found"})
        
    elif tool_name == "get_supplier_info":
        result = REALISTIC_DATA["suppliers"].get(param_value, {"error": "Supplier not found"})
        
    elif tool_name == "get_contact_details":
        result = REALISTIC_DATA["contacts"].get(param_value, {"error": "Contact not found"})
        
    elif tool_name == "get_territory_info":
        result = REALISTIC_DATA["territories"].get(param_value, {"error": "Territory not found"})
        
    elif tool_name == "get_manager_info":
        result = REALISTIC_DATA["managers"].get(param_value, {"error": "Manager not found"})
        
    else:
        result = {"error": "Unknown tool"}
    
    return json.dumps(result)


# ============================================================
# TOOL DEFINITIONS
# ============================================================

CHAIN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_user",
            "description": "Find user account by email address",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "User's email address"}
                },
                "required": ["email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_orders",
            "description": "Get all orders for a user",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID from lookup_user"}
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Get detailed information about an order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID from get_user_orders"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Check inventory levels for a product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "Product ID from order details"}
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_supplier_info",
            "description": "Get information about a supplier",
            "parameters": {
                "type": "object",
                "properties": {
                    "supplier_id": {"type": "string", "description": "Supplier ID from inventory"}
                },
                "required": ["supplier_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_contact_details",
            "description": "Get contact person details for a supplier",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_id": {"type": "string", "description": "Contact ID from supplier info"}
                },
                "required": ["contact_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_territory_info",
            "description": "Get information about a sales territory",
            "parameters": {
                "type": "object",
                "properties": {
                    "territory_id": {"type": "string", "description": "Territory ID from contact details"}
                },
                "required": ["territory_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_manager_info",
            "description": "Get information about a territory manager",
            "parameters": {
                "type": "object",
                "properties": {
                    "manager_id": {"type": "string", "description": "Manager ID from territory info"}
                },
                "required": ["manager_id"]
            }
        }
    }
]


# ============================================================
# TEST 1: CHAIN DEPTH LIMIT
# ============================================================

def test_chain_depth():
    """
    Test how deep a tool chain can go before model loses track.
    
    Chain: email → user_id → order_id → product_id → supplier_id → contact_id → territory_id → manager_id
    """
    print("\n" + "="*60)
    print("TEST 1: CHAIN DEPTH LIMIT")
    print("="*60)
    
    conversation = []
    
    # Chain steps with realistic queries
    chain_steps = [
        ("lookup_user", "Look up the user with email sarah.chen@techcorp.com"),
        ("get_user_orders", "Get all orders for this user"),
        ("get_order_details", "Get details of the first order"),
        ("check_inventory", "Check inventory for the first product in that order"),
        ("get_supplier_info", "Get the supplier information for this product"),
        ("get_contact_details", "Get the contact details for this supplier"),
        ("get_territory_info", "Get the territory information for this contact"),
        ("get_manager_info", "Get the manager information for this territory"),
    ]
    
    chain_history = []
    
    for step_num, (expected_tool, query) in enumerate(chain_steps, 1):
        print(f"\n--- Step {step_num}: {expected_tool} ---")
        print(f"Query: {query}")
        
        conversation.append({"role": "user", "content": query})
        
        # Get model response
        response = call_api(conversation, CHAIN_TOOLS)
        calls = extract_tool_calls(response)
        
        print(f"Calls: {calls}")
        
        if not calls:
            print(f"✗ FAILED - No tool call made")
            print(f"Chain broke at depth {step_num}")
            return step_num - 1, chain_history
        
        # Check if correct tool was called
        correct_tool = expected_tool in calls[0]
        print(f"Correct tool: {'✓' if correct_tool else '✗'}")
        
        if not correct_tool:
            print(f"✗ FAILED - Wrong tool called")
            print(f"Chain broke at depth {step_num}")
            return step_num - 1, chain_history
        
        # Execute tool
        conversation.append({"role": "assistant", "content": response})
        
        tool_result = mock_tool_execution(calls[0])
        print(f"Result: {tool_result[:100]}...")
        
        conversation.append({"role": "tool", "content": tool_result})
        chain_history.append({
            "step": step_num,
            "tool": expected_tool,
            "call": calls[0],
            "result": tool_result
        })
        
        # Get model's interpretation
        interpretation = call_api(conversation, CHAIN_TOOLS)
        print(f"Model interpretation: {interpretation[:80]}...")
        conversation.append({"role": "assistant", "content": interpretation})
    
    print(f"\n✓ SUCCESS - Chain completed all {len(chain_steps)} steps")
    return len(chain_steps), chain_history


# ============================================================
# TEST 2: HISTORY RETENTION LIMIT
# ============================================================

def test_history_retention(chain_history: List[Dict]):
    """
    Test how far back the model can recall information from tool results.
    """
    print("\n" + "="*60)
    print("TEST 2: HISTORY RETENTION LIMIT")
    print("="*60)
    
    if not chain_history:
        print("No chain history to test")
        return 0
    
    # Build conversation from chain history
    conversation = []
    for item in chain_history:
        conversation.append({"role": "user", "content": f"Execute step {item['step']}"})
        conversation.append({"role": "assistant", "content": f"<|tool_call_start|>[{item['call']}]<|tool_call_end|>"})
        conversation.append({"role": "tool", "content": item['result']})
        conversation.append({"role": "assistant", "content": f"Step {item['step']} completed"})
    
    # Test recall of various steps
    recall_tests = [
        (1, "What was the email address from step 1?", "sarah.chen@techcorp.com"),
        (1, "What was the user_id from step 1?", "usr_8k2m9p4"),
        (2, "What was the first order_id from step 2?", "ord_x9j2k1"),
        (3, "What was the first product_id from step 3?", "prod_wireless_kb"),
        (4, "What was the supplier_id from step 4?", "sup_logitech"),
    ]
    
    successful_recalls = 0
    
    for step_ref, query, expected_value in recall_tests:
        if step_ref > len(chain_history):
            continue
            
        print(f"\n--- Recall Test: Step {step_ref} ---")
        print(f"Query: {query}")
        print(f"Expected to mention: {expected_value}")
        
        conversation.append({"role": "user", "content": query})
        response = call_api(conversation, CHAIN_TOOLS)
        
        print(f"Response: {response[:100]}...")
        
        # Check if expected value is in response
        if expected_value.lower() in response.lower():
            print(f"✓ SUCCESS - Correctly recalled")
            successful_recalls += 1
        else:
            print(f"✗ FAILED - Could not recall")
        
        conversation.append({"role": "assistant", "content": response})
    
    recall_rate = (successful_recalls / len(recall_tests)) * 100
    
    print(f"\n--- Recall Summary ---")
    print(f"Successful recalls: {successful_recalls}/{len(recall_tests)} ({recall_rate:.0f}%)")
    
    return successful_recalls


# ============================================================
# MAIN
# ============================================================

def main():
    """Run both chain tests"""
    print("="*60)
    print("TOOL CHAINING LIMITS - REALISTIC DATA")
    print("="*60)
    
    try:
        # Test 1: Chain depth
        max_depth, chain_history = test_chain_depth()
        
        # Test 2: History retention
        successful_recalls = test_history_retention(chain_history)
        
        # Summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Maximum Chain Depth: {max_depth} steps")
        print(f"History Retention: {successful_recalls}/5 recalls successful")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
