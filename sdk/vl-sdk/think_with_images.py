#!/usr/bin/env python3
# coding: utf-8

"""
Cookbook: Think with Images (Local Version)

This script demonstrates how to use Qwen3-VL with image zoom-in and search capabilities
using a local LLM server instead of DashScope API.

All dependencies will be auto-installed if missing.
"""

import subprocess
import sys
import os

# Auto-install function
def install_package(package):
    """Install a package if it's not already installed"""
    try:
        __import__(package.split('[')[0])  # Handle packages like qwen-agent[gui]
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", package])
        print(f"✓ {package} installed successfully")

# Install required packages
print("="*70)
print("Checking and installing dependencies...")
print("="*70)

required_packages = [
    "qwen-agent",
    "requests",
    "pillow",
]

for package in required_packages:
    install_package(package)

print("\n✓ All dependencies installed!\n")

# Now import everything
from qwen_agent.agents import Assistant
from qwen_agent.utils.output_beautify import multimodal_typewriter_print
import requests
from PIL import Image
from io import BytesIO

# ============================================================================
# CONFIGURATION
# ============================================================================

# Local LLM Configuration
LOCAL_MODEL_ID = 'Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm'
LOCAL_API_BASE = 'http://localhost:8080/v1'

llm_cfg = {
    'model_type': 'qwenvl_oai',
    'model': LOCAL_MODEL_ID,
    'model_server': LOCAL_API_BASE,
    'api_key': 'EMPTY',
    'generate_cfg': {
        "top_p": 0.8,
        "top_k": 20,
        "temperature": 0.7,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5
    }
}

# ============================================================================
# EXAMPLE 1: Zoom-in Assistant
# ============================================================================

def example_1_zoom_in_assistant():
    """
    Example 1: Zoom-in Assistant
    
    Creates an agent capable of:
    - Analyzing images
    - Zooming in on specific regions (using image_zoom_in_tool)
    - Deep visual analysis
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Zoom-in Assistant")
    print("="*70)
    print("\nThis example demonstrates visual analysis with zoom-in capability.")
    print("The agent can zoom into specific regions to get more details.\n")
    
    analysis_prompt = """Your role is that of a research assistant specializing in visual information. Answer questions about images by looking at them closely and then using research tools. Please follow this structured thinking process and show your work.

Start an iterative loop for each question:

- **First, look closely:** Begin with a detailed description of the image, paying attention to the user's question. List what you can tell just by looking, and what you'll need to look up.
- **Next, find information:** Use a tool to research the things you need to find out.
- **Then, review the findings:** Carefully analyze what the tool tells you and decide on your next action.

Continue this loop until your research is complete.

To finish, bring everything together in a clear, synthesized answer that fully responds to the user's question."""

    tools = ['image_zoom_in_tool']
    
    print("Initializing agent with zoom-in capability...")
    agent = Assistant(
        llm=llm_cfg,
        function_list=tools,
        system_message=analysis_prompt,
    )
    
    # Example with a sample image URL
    # You can replace this with your own image URL or local file path
    test_image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    
    messages = [
        {
            "role": "user", 
            "content": [
                {"image": test_image_url},
                {"text": "What can you see in this image? Please analyze it in detail."}
            ]
        }
    ]
    
    print(f"\nImage URL: {test_image_url}")
    print("Question: What can you see in this image? Please analyze it in detail.")
    print("\n" + "-"*70)
    print("AGENT RESPONSE:")
    print("-"*70 + "\n")
    
    response_plain_text = ''
    try:
        for ret_messages in agent.run(messages):
            response_plain_text = multimodal_typewriter_print(ret_messages, response_plain_text)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPossible issues:")
        print("1. Local server might not support image_zoom_in_tool function")
        print("2. Image URL might not be accessible")
        print("3. Model might not support vision capabilities")
        return None
    
    print("\n" + "="*70 + "\n")
    return response_plain_text


# ============================================================================
# EXAMPLE 2: Multi-functional Assistant (Zoom + Search)
# ============================================================================

def example_2_multi_functional():
    """
    Example 2: Multi-functional Assistant
    
    Creates an agent with both:
    - image_zoom_in_tool (zoom into regions)
    - image_search (search for similar images)
    
    Note: image_search requires SERPER_API_KEY and SERPAPI_IMAGE_SEARCH_KEY
    environment variables to be set. If not available, this example will
    use zoom-in only.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Multi-functional Assistant (Zoom + Search)")
    print("="*70)
    
    # Check if search API keys are available
    has_search_keys = (
        os.environ.get('SERPER_API_KEY') and 
        os.environ.get('SERPAPI_IMAGE_SEARCH_KEY')
    )
    
    if not has_search_keys:
        print("\n⚠ WARNING: Search API keys not found!")
        print("To enable image search functionality, set these environment variables:")
        print("  export SERPER_API_KEY='your_key_here'")
        print("  export SERPAPI_IMAGE_SEARCH_KEY='your_key_here'")
        print("\nProceeding with zoom-in only...\n")
        tools = ['image_zoom_in_tool']
    else:
        print("\n✓ Search API keys found. Enabling both zoom and search.\n")
        tools = ['image_zoom_in_tool', 'image_search']
    
    print("Initializing multi-functional agent...")
    agent = Assistant(
        llm=llm_cfg,
        function_list=tools,
        system_message='Use tools to answer questions about images. Analyze carefully and use zoom or search as needed.',
    )
    
    # Example with a logo image
    test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/320px-Google_2015_logo.svg.png"
    
    messages = [{
        'role': 'user',
        'content': [
            {'image': test_image_url},
            {'text': 'Describe this logo in detail. What company does it belong to?'}
        ]
    }]
    
    print(f"Image URL: {test_image_url}")
    print("Question: Describe this logo in detail. What company does it belong to?")
    print("\n" + "-"*70)
    print("AGENT RESPONSE:")
    print("-"*70 + "\n")
    
    response_plain_text = ''
    try:
        for ret_messages in agent.run(messages):
            response_plain_text = multimodal_typewriter_print(ret_messages, response_plain_text)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPossible issues:")
        print("1. Local server might not support these tools")
        print("2. Image URL might not be accessible")
        print("3. Search API keys might be invalid (if search enabled)")
        return None
    
    print("\n" + "="*70 + "\n")
    return response_plain_text


# ============================================================================
# EXAMPLE 3: Custom Image Analysis
# ============================================================================

def example_3_custom_image(image_path_or_url, question):
    """
    Example 3: Analyze your own image
    
    Args:
        image_path_or_url: Local file path or URL to an image
        question: Your question about the image
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Image Analysis")
    print("="*70)
    
    tools = ['image_zoom_in_tool']
    
    agent = Assistant(
        llm=llm_cfg,
        function_list=tools,
        system_message='Analyze images carefully and answer questions about them.',
    )
    
    messages = [{
        'role': 'user',
        'content': [
            {'image': image_path_or_url},
            {'text': question}
        ]
    }]
    
    print(f"\nImage: {image_path_or_url}")
    print(f"Question: {question}")
    print("\n" + "-"*70)
    print("AGENT RESPONSE:")
    print("-"*70 + "\n")
    
    response_plain_text = ''
    try:
        for ret_messages in agent.run(messages):
            response_plain_text = multimodal_typewriter_print(ret_messages, response_plain_text)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None
    
    print("\n" + "="*70 + "\n")
    return response_plain_text


# ============================================================================
# TEST CONNECTION
# ============================================================================

def test_connection():
    """Test if the local LLM server is accessible"""
    print("\n" + "="*70)
    print("Testing Local LLM Connection")
    print("="*70)
    
    try:
        response = requests.get(f"{LOCAL_API_BASE.replace('/v1', '')}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"\n✓ Server is accessible at {LOCAL_API_BASE}")
            
            # Check if our model is available
            model_found = False
            if 'data' in models:
                print("\nAvailable models:")
                for model in models['data']:
                    model_id = model.get('id', 'unknown')
                    if model_id == LOCAL_MODEL_ID:
                        print(f"  ✓ {model_id} (target model)")
                        model_found = True
                    else:
                        print(f"    {model_id}")
            
            if model_found:
                print(f"\n✓ Target model '{LOCAL_MODEL_ID}' is available!")
                return True
            else:
                print(f"\n⚠ Warning: Model '{LOCAL_MODEL_ID}' not found in server")
                print("Please update LOCAL_MODEL_ID in the script to match an available model.")
                return False
        else:
            print(f"\n✗ Server returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Cannot connect to server at {LOCAL_API_BASE}")
        print("\nPlease ensure:")
        print("1. Your local LLM server is running")
        print("2. The server is accessible at http://localhost:8080")
        print("3. The /v1/models endpoint is available")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*70)
    print("Think with Images - Local LLM Version")
    print("="*70)
    print(f"Model: {LOCAL_MODEL_ID}")
    print(f"Server: {LOCAL_API_BASE}")
    print("="*70)
    
    # Test connection first
    if not test_connection():
        print("\n" + "="*70)
        print("CONNECTION FAILED - Cannot proceed")
        print("="*70)
        return
    
    print("\n" + "="*70)
    print("Ready to run examples!")
    print("="*70)
    
    # Run examples
    print("\nRunning examples... (This may take a few minutes)\n")
    
    # Example 1: Zoom-in assistant
    try:
        example_1_zoom_in_assistant()
    except Exception as e:
        print(f"\nExample 1 failed: {e}")
    
    # Example 2: Multi-functional (you can uncomment this)
    try:
        example_2_multi_functional()
    except Exception as e:
        print(f"\nExample 2 failed: {e}")
    
    # Example 3: Custom image (uncomment and modify as needed)
    # try:
    #     example_3_custom_image(
    #         image_path_or_url="your_image.jpg",
    #         question="What is in this image?"
    #     )
    # except Exception as e:
    #     print(f"\nExample 3 failed: {e}")
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nTo run custom examples:")
    print("1. Edit this script")
    print("2. Uncomment example_2 or example_3")
    print("3. Modify image URLs and questions as needed")
    print("="*70 + "\n")


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

def print_usage():
    """Print usage instructions"""
    usage = """
═══════════════════════════════════════════════════════════════════════
USAGE INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════

This script demonstrates Qwen-Agent with visual capabilities using a
local LLM server.

PREREQUISITES:
--------------
1. Local LLM server running at http://localhost:8080/v1
2. Model loaded: Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm
   (or update LOCAL_MODEL_ID in the script)

RUN THE SCRIPT:
--------------
python think_with_images_local.py

The script will:
1. Auto-install missing dependencies
2. Test connection to your local server
3. Run Example 1 (zoom-in assistant)

CUSTOMIZE:
----------
Edit the main() function to:
- Uncomment example_2_multi_functional() for search+zoom
- Uncomment example_3_custom_image() for your own images
- Modify image URLs and questions

EXAMPLE 3 - YOUR OWN IMAGE:
--------------------------
example_3_custom_image(
    image_path_or_url="/path/to/your/image.jpg",
    question="What is this?"
)

SEARCH FUNCTIONALITY:
--------------------
For image search in Example 2, set these before running:

export SERPER_API_KEY='your_serper_key'
export SERPAPI_IMAGE_SEARCH_KEY='your_serpapi_key'

Without these keys, only zoom-in will work.

CONFIGURATION:
--------------
Edit these at the top of the script:

LOCAL_MODEL_ID = 'your-model-id'
LOCAL_API_BASE = 'http://your-server:port/v1'

TROUBLESHOOTING:
----------------
If you get errors:
1. Check server is running: curl http://localhost:8080/v1/models
2. Verify model ID matches exactly
3. Ensure model supports vision (image input)
4. Check image URLs are accessible
5. Look at server logs for errors

NOTES:
------
- bbox_2d for image_zoom_in_tool uses relative coords [0, 1000]
- Not all local models support function calling (zoom/search)
- Some features require specific model capabilities

═══════════════════════════════════════════════════════════════════════
"""
    print(usage)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        print_usage()
    else:
        main()