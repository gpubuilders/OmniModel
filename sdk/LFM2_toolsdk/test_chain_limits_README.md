# Test Chain Limits

## Overview

Test Chain Limits is a comprehensive testing suite that evaluates two critical limitations of tool chaining in LLMs: chain depth limits and history retention capabilities. It uses realistic mock data to simulate a customer support workflow, testing how deep tool chains can go before the model loses track, and how well it can recall information from earlier steps.

## Key Features

- **Chain Depth Testing**: Tests how many sequential tools can be chained before model loses track
- **History Retention Testing**: Tests how far back the model can recall information from tool results
- **Realistic Data**: Uses realistic customer and product data to avoid pattern detection
- **Comprehensive Workflow**: Simulates a real customer support workflow from user lookup to manager info
- **Multiple Validation Points**: Validates both tool execution and result interpretation

## Architecture

### Core Components

- `REALISTIC_DATA`: Comprehensive mock data for users, orders, products, suppliers, etc.
- `call_api()`: Makes API calls to the LLM
- `extract_tool_calls()`: Parses tool calls from model responses
- `mock_tool_execution()`: Executes tools with realistic mock data
- `test_chain_depth()`: Tests maximum chain depth
- `test_history_retention()`: Tests history recall capabilities

### Mock Data Structure

The test uses an interconnected data model:
- Users → Orders → Order Details → Products → Inventory → Suppliers → Contacts → Territories → Managers
- Contains realistic customer information, order histories, product details, supplier data, and organizational hierarchies

### Tool Chain Workflow

The test follows this 8-step chain:
1. `lookup_user`: Find user by email
2. `get_user_orders`: Get orders for user
3. `get_order_details`: Get details of an order
4. `check_inventory`: Check product inventory
5. `get_supplier_info`: Get supplier information
6. `get_contact_details`: Get contact person
7. `get_territory_info`: Get territory information
8. `get_manager_info`: Get territory manager

## Tests

### Test 1: Chain Depth Limit
- Tests sequential tool execution in the customer support workflow
- Validates that each step calls the correct tool with correct parameters
- Measures how many steps the model can execute before breaking the chain
- Expected chain: email → user_id → order_id → product_id → supplier_id → contact_id → territory_id → manager_id

### Test 2: History Retention Limit
- Tests the model's ability to recall information from earlier steps
- Asks specific questions about data from previous tool results
- Measures recall accuracy across different step distances
- Tests recall of 5 different data points from the chain

## Configuration

- `BASE_URL`: Local LLM endpoint (default: "http://localhost:8080/v1")
- `MODEL_NAME`: Model to use for testing (default: "LFM2-1.2B-Tool-Q4_K_M-cuda")

## Output

The test provides detailed output for both tests:
- Step-by-step execution with tool calls and results
- Success/failure indicators for each step
- Maximum reliable chain depth
- History retention success rate
- Final summary with both metrics

## Results

Based on test execution:
- **Maximum Chain Depth**: 4 steps (breaks at step 5 when calling wrong tool)
- **History Retention**: 2/5 recalls successful (40% success rate)
- The chain breaks when the model calls an incorrect tool in the sequence

## Limitations

- Uses mock data instead of real services
- Tests with specific customer support workflow only
- Results depend on model's ability to follow the sequence
- History retention test may be affected by tool call patterns

## Dependencies

- `requests` for API communication
- `json` for data serialization
- `typing` for type hints
- `re` for pattern matching
- `random` for data variation