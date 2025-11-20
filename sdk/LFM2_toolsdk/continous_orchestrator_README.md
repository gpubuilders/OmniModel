# Continuous Orchestrator

## Overview

The Continuous Orchestrator is a production-ready tool chain orchestrator designed to run continuously while being aware of discovered limits. It manages tool chains and conversation history with awareness of depth limits, history retention, and recall capabilities.

## Key Features

- **Chain State Management**: Tracks conversation state with limit awareness
- **History Compression**: Automatically compresses old conversation history to stay within limits
- **Tool Result Caching**: Caches recent tool results for quick recall
- **Depth Limit Awareness**: Resets chain depth when limits are reached to prevent infinite loops
- **Continuous Operation**: Processes queries continuously from a stream

## Configuration

The orchestrator includes the following configurable parameters:

- `MAX_CHAIN_DEPTH`: Maximum tool chain depth before reset (default: 8)
- `MAX_HISTORY_MESSAGES`: Maximum messages before summarization (default: 20)
- `RECALL_SAFE_DEPTH`: Steps back that model can reliably recall (default: 5)
- `BASE_URL`: API endpoint base URL (default: "http://localhost:8080/v1")
- `MODEL_NAME`: Model to use for processing (default: "LFM2-1.2B-Tool-Q4_K_M-cuda")

## Architecture

### ChainState Class
Manages conversation state with:
- Conversation history tracking
- Chain depth counters
- Tool results caching
- History compression functionality

### ToolExecutor Class
Executes tools with realistic mock data for:
- User lookups
- Order retrieval
- Inventory checks
- Supplier information
- Contact details
- Territory information
- Manager information

### ToolChainOrchestrator Class
Main orchestrator that:
- Processes queries with tool chaining
- Manages state and limits
- Handles multiple iterations
- Prevents infinite loops

## Use Cases

### Customer Support Agent
Handles customer inquiries by looking up accounts and providing relevant information:
```python
support_agent = CustomerSupportAgent(orchestrator)
result = support_agent.handle_order_inquiry("sarah.chen@techcorp.com", "What's the status of my recent order?")
```

### Inventory Monitor
Monitors inventory levels and supplier information:
```python
monitor = InventoryMonitor(orchestrator)
result = monitor.check_low_stock_suppliers()
```

## Performance Results

Based on test runs:
- **Maximum Chain Depth**: 4 steps before chain breaks
- **History Retention**: 40% recall success rate
- **Successful recalls**: 2 out of 5 tested values
- **Rate limiting**: Includes 1-second delays between queries to prevent API overload

## Output Analysis

The orchestrator outputs detailed logs showing:
1. Chain depth at each step
2. Tool calls executed
3. Results of each operation
4. History compression events
5. Chain reset notifications when limits are exceeded

## Limitations

- Chain depth is limited to prevent infinite loops
- History retention decreases recall accuracy over time
- Model may occasionally call incorrect tools
- Requires a running API endpoint to function
- Mock data only - not connected to real systems

## Dependencies

- `requests` for API communication
- `json` for data serialization
- `time` for rate limiting
- `typing` for type hints
- `datetime` for timestamping
- `enum` for state management
- `re` for pattern matching