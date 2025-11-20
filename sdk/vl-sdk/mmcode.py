#!/usr/bin/env python
# coding: utf-8

# # Multimodal Coding Demo (Local Version)
# This notebook demonstrates three key capabilities of Qwen3-VL:
#
# 1. **Image to HTML**: Convert screenshots or sketches into functional HTML code
# 2. **Chart to Code**: Analyze chart images and generate corresponding plotting code
# 3. **Multimodal Coding Challenges**: Solve programming problems that require visual understanding
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
    from IPython.display import HTML, display as jupyter_display, Markdown
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
    "matplotlib",
    "numpy",
    "playwright",
]

for package in required_packages:
    install_package(package)

print("\n✓ All dependencies installed!\n")

# Optional: Only uncomment if you want to run local inference (not API-based)
# from transformers import AutoProcessor, AutoModelForVision2Seq
# from qwen_vl_utils import process_vision_info
# model_path = "path/to/local/model"
# processor = AutoProcessor.from_pretrained(model_path)
# model, output_loading_info = AutoModelForVision2Seq.from_pretrained(
#     model_path,
#     torch_dtype="auto",
#     device_map="auto",
#     output_loading_info=True
# )
# print("output_loading_info", output_loading_info)


import json
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def encode_image(image_path):
    """Encode image to base64 for OpenAI API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def convert_base64_to_pil_image(base64_str: str) -> Image:
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image

def show_image(image_path, max_width=1000):
    """Display image in notebook with size control"""
    if os.path.exists(image_path):
        img = Image.open(image_path)
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height))
        display(img)
    else:
        print(f"Image file {image_path} does not exist. Please download sample images to run this example.")

def show_pil_image(pil_image, max_width=1000):
    """Display PIL image in notebook with size control"""
    img = pil_image.copy()
    if img.width > max_width:
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_width, new_height))
    display(img)

print("✅ Setup complete!")

def extract_last_code_block(text):
    """Extract the last named markdown code block from the text"""
    import re
    code_blocks = re.findall(r"```(?:python|html)(.*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()
    return None


# API Inference function using local OpenAI-compatible server

def sketch_to_html_with_local_api(image_path):
    """Convert sketch to HTML using local Qwen3-VL model"""
    base64_image = encode_image(image_path)
    
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_HTTP_API_URL'),
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    # Feel free to modify the prompt to get different styles of HTML
                    "text": """Analyze this screenshot and convert it to clean, functional and modern HTML code. """
                },
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=LOCAL_MODEL_ID,
            messages=messages,
            max_tokens=16384,
            temperature=0.8,
        )

        response_text = response.choices[0].message.content
        print(response_text)
        code = extract_last_code_block(response_text)
        if code is None:
            raise ValueError("No code block found in the response.")
        return code
    except Exception as e:
        print(f"Error during API call: {e}")
        raise


def chart_to_matplotlib_with_local_api(image_path):
    """Convert chart to matplotlib code using local Qwen3-VL model"""
    base64_image = encode_image(image_path)
    
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_HTTP_API_URL'),
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Convert this chart image to Python matplotlib code that can reproduce the chart."""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=LOCAL_MODEL_ID,
            messages=messages,
            max_tokens=16384,
            temperature=0.8
        )

        response_text = response.choices[0].message.content
        code = extract_last_code_block(response_text)
        if code is None:
            raise ValueError("No code block found in the response.")
        return code
    except Exception as e:
        print(f"Error during API call: {e}")
        raise


def solve_mmcode_problem_with_local_api(problem):
    """Solve an MMCode problem using local Qwen3-VL with correct image interleaving"""
    if not problem:
        return "No problem selected."

    # Generate prompt following MMCode format
    prompt = (
        "You are required to solve a programming problem. "
        + "Please enclose your code inside a ```python``` block. "
        + " Do not write a main() function. If Call-Based format is used, return the result in an appropriate place instead of printing it.\n\n"
        + generate_prompt(problem)
    )

    # Convert all images to base64 (images are guaranteed to exist)
    base64_images = []
    for pil_image in problem['images']:
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        base64_images.append(img_base64)

    interleaved_messages = []

    # Split the prompt and interleave text and images (following MMCode's approach)
    import re
    segments = re.split(r"!\[image\]\(.*?\)", prompt)
    for i, segment in enumerate(segments):
        # Text
        if len(segment) > 0:
            interleaved_messages.append({"type": "text", "text": segment})
        # Image
        if i < len(base64_images):
            interleaved_messages.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_images[i]}",
                    },
                }
            )

    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_HTTP_API_URL'),
    )

    try:
        response = client.chat.completions.create(
            model=LOCAL_MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a professional programming contester trying to solve algorithmic problems. The problems come with a description and some images, and you should write a Python solution.",
                        }
                    ],
                },
                {"role": "user", "content": interleaved_messages},
            ],
            max_tokens=16384,
            temperature=0.1
        )
        
        raw_response = response.choices[0].message.content
        code = extract_last_code_block(raw_response)
        if code is None:
            raise ValueError("No code block found in the response.")
        return code
    except Exception as e:
        return f"Error generating solution: {e}"


def generate_prompt(problem):
    """Generate prompt following MMCode's format"""
    prompt = "\nQUESTION:\n"
    prompt += problem["question"]
    starter_code = problem["starter_code"] if len(problem.get("starter_code", [])) > 0 else None
    try:
        input_output = json.loads(problem["input_output"])
        fn_name = None if not input_output.get("fn_name") else input_output["fn_name"]
    except ValueError:
        fn_name = None

    if (not fn_name) and (not starter_code):
        call_format = "\nPlease write your code using Standard IO, i.e. input() and print()."
        prompt += call_format
    else:
        call_format = "\nPlease write your code using Call-Based format."
        prompt += call_format

    if starter_code:
        prompt += "\nThe starter code is provided as below. Please finish the code.\n" + starter_code + "\n"

    prompt += "\nANSWER:\n"
    return prompt


def example_1_image_to_html():
    """Example 1: Image-to-HTML Conversion"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Image-to-HTML Conversion")
    print("="*70)

    # Use local image path
    input_image_path = "assets/multimodal_coding/screenshot_demo.png"
    
    print(f"Input image path: {input_image_path}")
    
    if os.path.exists(input_image_path):
        print("Input image:")
        show_image(input_image_path)
        
        try:
            # Generate HTML from sketch
            html_code = sketch_to_html_with_local_api(input_image_path)
            print("\n" + "="*50)
            print("Generated HTML:")
            print("="*50)
            print(html_code)

            # Save the generated HTML to a file
            output_html_path = "image2code_output.html"
            with open(output_html_path, "w") as f:
                f.write(html_code)
            
            print(f"\n✅ Generated HTML saved to {output_html_path}")
        except Exception as e:
            print(f"Error in example: {e}")
    else:
        print(f"Image file {input_image_path} does not exist. Please download sample images to run this example.")
    
    print("="*70 + "\n")


def example_2_chart_to_code():
    """Example 2: Chart-to-Code"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Chart-to-Code")
    print("="*70)

    # Use local chart path
    chart_path = "assets/multimodal_coding/chart2code_input.png"
    
    print(f"Input chart path: {chart_path}")
    
    if os.path.exists(chart_path):
        print("Input chart:")
        show_image(chart_path)
        
        try:
            # Generate matplotlib code from chart
            matplotlib_code = chart_to_matplotlib_with_local_api(chart_path)
            print("\n" + "="*50)
            print("Generated Matplotlib Code:")
            print("="*50)
            print(matplotlib_code)

            # Execute the generated matplotlib code
            # =====================================
            ALLOW_DANGEROUS_CODE = False # Set to True to enable execution
            # =====================================
            print("Executing generated code...")
            try:
                if not ALLOW_DANGEROUS_CODE:
                    raise RuntimeError("Execution of generated code is disabled for safety. Set ALLOW_DANGEROUS_CODE = True to enable.")
                # Execute the generated code
                exec(matplotlib_code)
                plt.show()
            except Exception as e:
                print(f"Error executing code: {e}")
                print("You may need to adjust the generated code slightly.")
        except Exception as e:
            print(f"Error in example: {e}")
    else:
        print(f"Image file {chart_path} does not exist. Please download sample images to run this example.")
    
    print("="*70 + "\n")


def download_mmcode_testset():
    """Download MMCode test set from HuggingFace or use local file if available"""
    url = "https://huggingface.co/datasets/likaixin/MMCode/resolve/main/mmcode_test.jsonl.gz?download=true"
    local_path = "mmcode_test.jsonl.gz"

    # Check if file already exists locally
    if os.path.exists(local_path):
        print(f"Using existing local MMCode test set: {local_path}")
        return local_path

    print("Downloading MMCode test set...")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded to {local_path}")
        return local_path
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def load_mmcode_problems(file_path):
    """Load problems from the downloaded JSONL file and convert images to PIL"""
    problems = []
    try:
        import gzip
        with gzip.open(file_path, 'rt') as f:
            for line in f:
                problem = json.loads(line)

                # Convert base64 images to PIL Images (images are guaranteed to exist)
                pil_images = []
                for img_base64 in problem['images']:
                    try:
                        pil_img = convert_base64_to_pil_image(img_base64)
                        pil_images.append(pil_img)
                    except Exception as e:
                        print(f"Warning: Failed to decode image: {e}")
                        continue
                problem['images'] = pil_images

                problems.append(problem)

        print(f"✅ Loaded {len(problems)} problems")
        return problems
    except Exception as e:
        print(f"❌ Failed to load problems: {e}")
        return []


# Interactive problem selection and display
def display_problem(problems, problem_index):
    """Display a specific MMCode problem with proper formatting and image-text interleaving"""
    if not problems or problem_index >= len(problems):
        print(f"❌ Invalid problem index. Available: 0-{len(problems)-1}")
        return None

    problem = problems[problem_index]

    print("="*60)
    print(f"PROBLEM {problem_index}: {problem.get('problem_id', 'Unknown')}")
    print("="*60)

    # Display problem statement with image-text interleaving
    print("\n📝 PROBLEM STATEMENT:")
    print("-" * 40)

    question_text = problem.get('question', 'No question found')

    # Get images (guaranteed to exist per user specification)
    images = problem.get('images', [])

    # Split the question text and interleave with images
    import re
    segments = re.split(r"!\[image\]\(.*?\)", question_text)

    for i, segment in enumerate(segments):
        # Print text segment
        if len(segment.strip()) > 0:
            print(segment.strip())

        # Display corresponding image if available
        if i < len(images):
            show_pil_image(images[i], max_width=500)
            print()  # Add spacing after image

    # Display starter code if available
    if problem.get('starter_code'):
        print("\n💻 STARTER CODE:")
        print("-" * 40)
        print(problem['starter_code'])

    # Display test cases info
    if problem.get('input_output'):
        try:
            io_data = json.loads(problem['input_output'])
            if 'inputs' in io_data and 'outputs' in io_data:
                print(f"\n🧪 TEST CASES: {len(io_data['inputs'])} available")
                print("-" * 40)
                # Show first test case as example
                if io_data['inputs'] and io_data['outputs']:
                    print(f"Example - Input: {io_data['inputs'][0]}")
                    print(f"Example - Output: {io_data['outputs'][0]}")
        except:
            pass

    return problem


def example_3_mmcode_problems():
    """Example 3: Multimodal Coding Challenges"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Multimodal Coding Challenges")
    print("="*70)

    # Download and load the test set
    test_file = download_mmcode_testset()
    if test_file is None:
        print("❌ Could not load MMCode test set. Please ensure mmcode_test.jsonl.gz is available.")
        print("="*70 + "\n")
        return

    problems = load_mmcode_problems(test_file)
    if not problems:
        print("❌ No problems loaded from the test set.")
        print("="*70 + "\n")
        return

    print(f"Total problems available: 0 to {len(problems)-1}")

    # Select a problem to work with
    # =====================================
    PROBLEM_INDEX = 0  # Change this to select different problems (0-263)
    # =====================================

    selected_problem = display_problem(problems, PROBLEM_INDEX)

    if selected_problem:
        print("\n🔄 Generating solution...")
        solution_code = solve_mmcode_problem_with_local_api(selected_problem)
        print("\n" + "="*50)
        print("🤖 GENERATED SOLUTION:")
        print("="*50)
        print(solution_code)

        # Save solution to file in output directory
        os.makedirs("output", exist_ok=True)
        solution_output_path = "output/mmcode_solution.txt"
        with open(solution_output_path, "w") as f:
            f.write(solution_code)

        print(f"\n✅ Solution saved to {solution_output_path}")

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
    print("Multimodal Coding with Local LLM")
    print("="*60)
    print(f"Model: {LOCAL_MODEL_ID}")
    print(f"Endpoint: {os.getenv('OPENAI_BASE_HTTP_API_URL')}")
    print("="*60)

    # Test API connection first
    if not test_api_connection():
        print("\nCannot proceed without API connection.")
        return

    print("\nRunning multimodal coding examples...")
    
    # Example 1: Image to HTML
    example_1_image_to_html()
    
    # Example 2: Chart to code
    example_2_chart_to_code()
    
    # Example 3: MMCode problems
    example_3_mmcode_problems()

    print("Done!")


if __name__ == "__main__":
    main()