# LFM2 Tool Calling - Correct Format

## The Solution

Your server requires the **modern OpenAI nested format** with both `"type"` and `"function"` fields.

### ❌ Old Format (doesn't work)
```json
{
  "tools": [
    {
      "type": "function",
      "name": "get_weather",
      "description": "Get current weather",
      "parameters": {...}
    }
  ]
}
```

### ✅ Correct Format (works!)
```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

## Key Differences

1. **Nested structure**: Tool definition goes inside a `"function"` object
2. **Two-level hierarchy**: 
   - Top level: `{"type": "function", "function": {...}}`
   - Inner level: `{"name": "...", "description": "...", "parameters": {...}}`

## Python Example

```python
# Correct way to define tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]

# Use in request
payload = {
    "model": "LFM2-8B-A1B-BF16-cuda",
    "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
    "tools": TOOLS
}
```

## Expected Response Format

The model will respond with tool calls in this format:

```
<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>
```

## Tool Response Format

When providing tool results back to the model:

```python
{
    "role": "tool",
    "content": '[{"temp": 22, "condition": "sunny"}]'
}
```

## Complete Conversation Example

```python
# Turn 1: User asks question
messages = [
    {"role": "user", "content": "What's the weather in Tokyo?"}
]

# Model responds with tool call
response = "<|tool_call_start|>[get_weather(city='Tokyo')]<|tool_call_end|>"

# Turn 2: Provide tool result
messages.append({"role": "assistant", "content": response})
messages.append({
    "role": "tool", 
    "content": '[{"temp": 22, "condition": "sunny"}]'
})

# Turn 3: Model interprets result
# Model will now generate a natural language response using the tool result
```

## Common Errors

1. **"Missing tool function"** → You're using flat format, need nested
2. **"Missing tool type"** → You're missing the outer `"type": "function"` field
3. **"Unsupported tool type"** → You're using wrong type value (must be "function")

## Testing

Run the fixed test suite:
```bash
python test_toolcall_fixed.py
```

This will test:
- Basic single tool call
- Tool visibility (1-8 tools)
- Sequential multi-tool calls
- Multi-turn conversations