#!/usr/bin/env python
# coding: utf-8

# ### Agent Function Call with Qwen3-VL (Local Version)
#
# This notebook demonstrates how to use Qwen3-VL's agent function call capabilities to interact with a mobile device. It showcases the model's ability to generate and execute actions based on user queries and visual context.
# This version has been modified to use a local LLM server instead of DashScope API.

import os
import sys
import subprocess
from openai import OpenAI
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Local LLM Server Configuration
os.environ['OPENAI_API_KEY'] = 'dummy-key'  # Local server might not validate this
os.environ['OPENAI_BASE_HTTP_API_URL'] = 'http://localhost:8080/v1'
LOCAL_MODEL_ID = 'Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm'

# Detect if running in Jupyter or terminal
try:
    get_ipython()
    IN_JUPYTER = True
    from IPython.display import Markdown, display as jupyter_display
except NameError:
    IN_JUPYTER = False

def display(content):
    """Display content - works in both Jupyter and terminal"""
    if IN_JUPYTER:
        from IPython.display import display as jupyter_display
        jupyter_display(content)
    else:
        if hasattr(content, 'save'):  # PIL Image
            print("[Image would be displayed here in Jupyter]")
            print(f"Image size: {content.size}")
        else:
            print(content)

def Markdown(text):
    """Markdown wrapper - works in both Jupyter and terminal"""
    if IN_JUPYTER:
        from IPython.display import Markdown as JupyterMarkdown
        return JupyterMarkdown(text)
    else:
        return text

def install_package(package):
    """Install a package if it's not already installed"""
    try:
        __import__(package.split('[')[0])
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", package])
        print(f"✓ {package} installed successfully")

# Check and install required packages
print("="*70)
print("Checking and installing dependencies...")
print("="*70)

required_packages = [
    "Pillow",
    "requests",
    "openai",
    "icecream",
    "matplotlib",
]

for package in required_packages:
    install_package(package)

print("\n✓ All dependencies installed!\n")

import os.path as osp
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from icecream import ic
import math
from PIL import Image, ImageDraw, ImageFont, ImageColor

def draw_point(image: Image.Image, point: list, color=None):
    from copy import deepcopy
    if isinstance(color, str):
        try:
            color = ImageColor.getrgb(color)
            color = color + (128,)
        except ValueError:
            color = (255, 0, 0, 128)
    else:
        color = (255, 0, 0, 128)

    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    radius = min(image.size) * 0.05
    x, y = point

    overlay_draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=color  # Red with 50% opacity
    )

    image = image.convert('RGBA')
    combined = Image.alpha_composite(image, overlay)

    return combined.convert('RGB')

def rescale_coordinates(point, width, height):
    point = [round(point[0]/999*width), round(point[1]/999*height)]
    return point


# API Inference function using local OpenAI-compatible server

import json
import base64

#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def inference_with_local_api(screenshot, instruction, history, system_prompt, model_id=LOCAL_MODEL_ID):
    """
    Perform inference using local OpenAI-compatible API server.

    Args:
        screenshot: Path to the screenshot image
        instruction: User instruction
        history: List of previous actions
        system_prompt: System prompt for the model
        model_id: Model ID to use for inference

    Returns:
        dict: Action to perform based on model's response
    """
    stage2_history = ''
    for idx, his in enumerate(history):
        stage2_history += 'Step ' + str(idx + 1) + ': ' + str(his.replace('\n', '').replace('"', '')) + '; '

    user_query = f"The user query: {instruction}.\nTask progress (You have done the following operation on the current device): {stage2_history}.\n"

    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_HTTP_API_URL'),
    )

    # The resolution of the device will be written into the system prompt.
    dummy_image = Image.open(screenshot)

    base64_image = encode_image(screenshot)
    # Build messages

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },

            ],
        }
    ]
    # print(json.dumps(messages, indent=4))
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
    )

    output_text = completion.choices[0].message.content
    # Qwen will perform action thought function call
    
    # Try to extract the function call from the response
    try:
        # Handle the XML-like tags properly - simplified approach
        if '"name":' in output_text and '"arguments":' in output_text:
            # Look for JSON object with name and arguments
            start_idx = output_text.find('{')
            end_idx = output_text.rfind('}') + 1
            json_str = output_text[start_idx:end_idx]
            action = json.loads(json_str)
        else:
            # Fallback for different output format
            action = {"error": "Could not parse action from response"}
    except json.JSONDecodeError:
        action = {"error": f"Could not parse JSON from response: {output_text}"}
    
    return action


def example_1_english_app():
    """Example 1: English App & Query with local API"""
    print("\n" + "="*70)
    print("EXAMPLE 1: English App & Query with Local API")
    print("="*70)

    system_prompt = """\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <functioncall></functioncall> XML tags:\n<functioncall>\n{"name": <function-name>, "arguments": <args-json-object>}\n</functioncall>\n\n# Response format\n\nResponse format for every step:\n1) Thought: one concise sentence explaining the next move (no multi-step reasoning).\n2) Action: a short imperative describing what to do in the UI.\n3) A single <functioncall>...</functioncall> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.\n\nRules:\n- Output exactly in the order: Thought, Action, <functioncall>.\n- Be brief: one sentence for Thought, one for Action.\n- Do not output anything else outside those three parts.\n- If finishing, use action=terminate in the tool call."""

    screenshot = "./assets/agent_function_call/mobile_en_example.png"
    instruction = "Search for Musk in X and go to his homepage to open the first post."

    history = [
        "I opened the X app from the home screen.",
    ]

    if os.path.exists(screenshot):
        print(f"Screenshot: {screenshot}")
        print(f"Instruction: {instruction}")
        
        try:
            action = inference_with_local_api(screenshot, instruction, history, system_prompt, LOCAL_MODEL_ID)
            print(f"\nGenerated action: {action}")

            # As an example, we visualize the "click" action by draw a green circle onto the image.
            if action.get('arguments', {}).get('action') == "click":
                dummy_image = Image.open(screenshot)
                coordinate = action['arguments']['coordinate']
                coordinate = rescale_coordinates(coordinate, dummy_image.width, dummy_image.height)
                dummy_image = draw_point(dummy_image, coordinate, color='green')
                display(dummy_image)
            else:
                dummy_image = Image.open(screenshot)
                display(dummy_image)
        except Exception as e:
            print(f"Error in example: {e}")
    else:
        print(f"Screenshot file {screenshot} does not exist. Please download sample images to run this example.")

    print("="*70 + "\n")


def example_2_chinese_app():
    """Example 2: Chinese App & Query with local API"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Chinese App & Query with Local API")
    print("="*70)

    system_prompt = """\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the top edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <functioncall></functioncall> XML tags:\n<functioncall>\n{"name": <function-name>, "arguments": <args-json-object>}\n</functioncall>\n\n# Response format\n\nResponse format for every step:\n1) Thought: one concise sentence explaining the next move (no multi-step reasoning).\n2) Action: a short imperative describing what to do in the UI.\n3) A single <functioncall>...</functioncall> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.\n\nRules:\n- Output exactly in the order: Thought, Action, <functioncall>.\n- Be brief: one sentence for Thought, one for Action.\n- Do not output anything else outside those three parts.\n- If finishing, use action=terminate in the tool call."""

    screenshot = "./assets/agent_function_call/mobile_zh_example.png"
    instruction = "Search for Musk in X and go to his homepage to open the first post."

    history = [
        "Opening the Bilibili app.",
        "Clicking on the search bar to start searching for Qwen-VL videos.",
        "Type 'Qwen-VL' into the search bar.",
        "Clicking the '搜索' button to initiate the search for Qwen-VL videos.",
    ]

    if os.path.exists(screenshot):
        print(f"Screenshot: {screenshot}")
        print(f"Instruction: {instruction}")
        
        try:
            action = inference_with_local_api(screenshot, instruction, history, system_prompt, LOCAL_MODEL_ID)
            print(f"\nGenerated action: {action}")

            # As an example, we visualize the "click" action by draw a green circle onto the image.
            if action.get('arguments', {}).get('action') == "click":
                dummy_image = Image.open(screenshot)
                coordinate = action['arguments']['coordinate']
                coordinate = rescale_coordinates(coordinate, dummy_image.width, dummy_image.height)
                dummy_image = draw_point(dummy_image, coordinate, color='green')
                display(dummy_image)
            else:
                dummy_image = Image.open(screenshot)
                display(dummy_image)
        except Exception as e:
            print(f"Error in example: {e}")
    else:
        print(f"Screenshot file {screenshot} does not exist. Please download sample images to run this example.")

    print("="*70 + "\n")


# Simple test function to verify API connection
def test_api_connection():
    """Test if the local API is accessible"""
    print("\n" + "="*60)
    print("Testing API Connection...")
    print("="*60)

    try:
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_HTTP_API_URL'),
        )

        # Try a simple text completion first
        test_messages = [
            {
                "role": "user",
                "content": "Hello! Please respond with 'API connection successful!'"
            }
        ]

        completion = client.chat.completions.create(
            model=LOCAL_MODEL_ID,
            messages=test_messages,
            max_tokens=50
        )

        response = completion.choices[0].message.content
        print(f"\n✓ API Connection Successful!")
        print(f"Model: {LOCAL_MODEL_ID}")
        print(f"Response: {response}\n")
        return True

    except Exception as e:
        print(f"\n✗ API Connection Failed!")
        print(f"Error: {e}\n")
        print("Please check:")
        print("  1. Is your local LLM server running at http://localhost:8080?")
        print("  2. Is the model ID correct: Qwen3-VL-8B-Instruct-AWQ-8bit-24k-vllm")
        print("  3. Try: curl http://localhost:8080/v1/models\n")
        return False


# Main execution function
def main():
    """
    Main function to run examples.
    """
    print("="*60)
    print("Mobile Agent with Local LLM")
    print("="*60)
    print(f"Model: {LOCAL_MODEL_ID}")
    print(f"Endpoint: {os.getenv('OPENAI_BASE_HTTP_API_URL')}")
    print("="*60)

    # Test API connection first
    if not test_api_connection():
        print("\nCannot proceed without API connection.")
        return

    print("\nRunning mobile agent examples...")
    
    # Example 1: English app
    example_1_english_app()
    
    # Example 2: Chinese app
    example_2_chinese_app()

    print("Done!")


if __name__ == "__main__":
    main()