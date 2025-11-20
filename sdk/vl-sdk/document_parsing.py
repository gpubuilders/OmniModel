#!/usr/bin/env python
# coding: utf-8

# ### Document Parsing with Qwen3-VL (Local Version)
#
# Welcome to this notebook, which showcases the powerful document parsing capabilities of our model. It can process any image and output its contents in various formats such as HTML, JSON, Markdown, and LaTeX. Notably, we introduce two unique Qwenvl formats:
#
# - Qwenvl HTML format, which adds positional information for each component to enable precise document reconstruction and manipulation.
# - Qwenvl Markdown format, which converts the overall image content into Markdown. In this format, all tables are represented in LaTeX with their corresponding coordinates indicated before each table, and images are replaced with coordinate-based placeholders for accurate positioning.
#
# This allows for highly detailed and flexible document parsing and reconstruction.
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
    "beautifulsoup4",
    "qwen-agent",
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


# Get Noto font
# !apt-get install fonts-noto-cjk

import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from bs4 import BeautifulSoup, Tag
import re

# Function to draw bounding boxes and text on images based on HTML content
def draw_bbox_html(image_path, full_predict):
    """
    可视化 Qwenvl HTML 的 data-bbox 框并展示文本，坐标为相对 0-1000。
    过滤规则：跳过 <ol>，仅绘制 <li> 子项和其它元素。
    """
    # 读取图片
    if image_path.startswith("http"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    width = image.width
    height = image.height

    soup = BeautifulSoup(full_predict, 'html.parser')
    elements_with_bbox = soup.find_all(attrs={'data-bbox': True})

    # 保留原过滤逻辑
    filtered_elements = []
    for el in elements_with_bbox:
        if el.name == 'ol':
            continue  # 跳过 <ol>
        elif el.name == 'li' and el.parent.name == 'ol':
            filtered_elements.append(el)  # 仅保留 <ol> 下的 <li>
        else:
            filtered_elements.append(el)

    # 字体兼容
    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 10)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)

    # 绘制框与文本
    for element in filtered_elements:
        bbox_str = element['data-bbox']
        text = element.get_text(strip=True)
        try:
            x1, y1, x2, y2 = map(int, bbox_str.split())
        except Exception:
            continue

        bx1 = int(x1 / 1000 * width)
        by1 = int(y1 / 1000 * height)
        bx2 = int(x2 / 1000 * width)
        by2 = int(y2 / 1000 * height)

        if bx1 > bx2:
            bx1, bx2 = bx2, bx1
        if by1 > by2:
            by1, by2 = by2, by1

        draw.rectangle([bx1, by1, bx2, by2], outline='red', width=2)
        draw.text((bx1, by2), text, fill='black', font=font)

    display(image)


# Function to draw bounding boxes on images based on Markdown content
def draw_bbox_markdown(image_path, md_content):
    """
    只可视化Markdown中的 <!-- Image/Table (x1, y1, x2, y2) --> 坐标框，坐标为相对0-1000
    Table 用绿色框，Image 用蓝色框。
    """
    if image_path.startswith("http"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    width = image.width
    height = image.height

    pattern = r"<!-- (Image|Table) \(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\) -->"
    matches = re.findall(pattern, md_content)
    draw = ImageDraw.Draw(image)
    for item in matches:
        typ, x1, y1, x2, y2 = item
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        bx1 = int(x1 / 1000 * width)
        by1 = int(y1 / 1000 * height)
        bx2 = int(x2 / 1000 * width)
        by2 = int(y2 / 1000 * height)
        if bx1 > bx2:
            bx1, bx2 = bx2, bx1
        if by1 > by2:
            by1, by2 = by2, by1
        color = 'blue' if typ == "Image" else 'red'
        draw.rectangle([bx1, by1, bx2, by2], outline=color, width=6)

    display(image)


# Function to clean and format HTML content
def clean_and_format_html(full_predict):
    soup = BeautifulSoup(full_predict, 'html.parser')

    # Regular expression pattern to match 'color' styles in style attributes
    color_pattern = re.compile(r'\bcolor:[^;]+;?')

    # Find all tags with style attributes and remove 'color' styles
    for tag in soup.find_all(style=True):
        original_style = tag.get('style', '')
        new_style = color_pattern.sub('', original_style)
        if not new_style.strip():
            del tag['style']
        else:
            new_style = new_style.rstrip(';')
            tag['style'] = new_style

    # Remove 'data-bbox' and 'data-polygon' attributes from all tags
    for attr in ["data-bbox", "data-polygon"]:
        for tag in soup.find_all(attrs={attr: True}):
            del tag[attr]

    classes_to_update = ['formula.machine_printed', 'formula.handwritten']
    # Update specific class names in div tags
    for tag in soup.find_all(class_=True):
        if isinstance(tag, Tag) and 'class' in tag.attrs:
            new_classes = [cls if cls not in classes_to_update else 'formula' for cls in tag.get('class', [])]
            tag['class'] = list(dict.fromkeys(new_classes))  # Deduplicate and update class names

    # Clear contents of divs with specific class names and rename their classes
    for div in soup.find_all('div', class_='image caption'):
        div.clear()
        div['class'] = ['image']

    classes_to_clean = ['music sheet', 'chemical formula', 'chart']
    # Clear contents and remove 'format' attributes of tags with specific class names
    for class_name in classes_to_clean:
        for tag in soup.find_all(class_=class_name):
            if isinstance(tag, Tag):
                tag.clear()
                if 'format' in tag.attrs:
                    del tag['format']

    # Manually build the output string
    output = []
    for child in soup.body.children:
        if isinstance(child, Tag):
            output.append(str(child))
            output.append('\n')  # Add newline after each top-level element
        elif isinstance(child, str) and not child.strip():
            continue  # Ignore whitespace text nodes
    complete_html = f"""```html\n<html><body>\n{" ".join(output)}</body></html>\n```"""
    return complete_html

# API Inference function using local OpenAI-compatible server

import base64

#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def inference_with_local_api(image_path, prompt, model_id=LOCAL_MODEL_ID, min_pixels=512*32*32, max_pixels=2048*32*32):
    """
    Perform inference using local OpenAI-compatible API server.

    Args:
        image_path: Path to image input (local file path)
        prompt: Text prompt for the model
        model_id: Model ID to use for inference
        min_pixels: Minimum pixels for image processing
        max_pixels: Maximum pixels for image processing

    Returns:
        str: Generated text response from the model
    """
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
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                    # PNG image:  f"data:image/png;base64,{base64_image}"
                    # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                    # WEBP image: f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error during API call: {e}")
        raise


def example_1_document_parsing_html():
    """Example 1: Document Parsing in QwenVL HTML Format"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Document Parsing in QwenVL HTML Format")
    print("="*70)

    img_url = "https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/Qwen3VL/demo/omni_parsing/179729.jpg"
    response = requests.get(img_url)
    img_name = os.path.basename(img_url)
    
    # Save the image temporarily
    image = Image.open(BytesIO(response.content))
    image.save(img_name)
    
    prompt = "qwenvl html"

    print(f"Image URL: {img_url}")
    print(f"Prompt: {prompt}")
    
    try:
        # Use local API for inference
        output = inference_with_local_api(img_name, prompt)
        print("\nResponse:", output)
        
        # Visualization
        print(f"Image size: {image.size[1]} x {image.size[0]}")
        draw_bbox_html(img_url, output)

        ordinary_html = clean_and_format_html(output)
        print("\nCleaned HTML:")
        print(ordinary_html)
    except Exception as e:
        print(f"Error in example: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(img_name):
            os.remove(img_name)

    print("="*70 + "\n")


def example_2_document_parsing_markdown():
    """Example 2: Document Parsing in QwenVL Markdown Format"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Document Parsing in QwenVL Markdown Format")
    print("="*70)

    img_url = "https://ofasys-multimodal-wlcb-3-toshanghai.oss-cn-shanghai.aliyuncs.com/Qwen3VL/demo/omni_parsing/120922.jpg"
    response = requests.get(img_url)
    img_name = os.path.basename(img_url)
    
    # Save the image temporarily
    image = Image.open(BytesIO(response.content))
    image.save(img_name)

    prompt = "qwenvl markdown"

    print(f"Image URL: {img_url}")
    print(f"Prompt: {prompt}")
    
    try:
        # Use local API for inference
        output = inference_with_local_api(img_name, prompt)
        print("\nResponse:", output)
        
        # Visualization
        print(f"Image size: {image.size[1]} x {image.size[0]}")
        draw_bbox_markdown(img_url, output)
    except Exception as e:
        print(f"Error in example: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(img_name):
            os.remove(img_name)

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
    print("Document Parsing with Local LLM")
    print("="*60)
    print(f"Model: {LOCAL_MODEL_ID}")
    print(f"Endpoint: {os.getenv('OPENAI_BASE_HTTP_API_URL')}")
    print("="*60)

    # Test API connection first
    if not test_api_connection():
        print("\nCannot proceed without API connection.")
        return

    print("\nRunning document parsing examples...")
    
    # Example 1: Document parsing in HTML format
    example_1_document_parsing_html()
    
    # Example 2: Document parsing in Markdown format
    example_2_document_parsing_markdown()

    print("Done!")


if __name__ == "__main__":
    main()