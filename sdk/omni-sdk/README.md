# Qwen3-Omni Local Cookbook Examples

This repository contains local versions of the Qwen3-Omni cookbook examples that use a local API endpoint instead of remote models.

## Overview

The original cookbook examples were designed to work with remote models. This repository provides local versions that connect to a local API endpoint at `http://localhost:8080/v1` using the Qwen3-Omni-10k model.

## Local API Setup

Make sure you have a local LLM server running at `http://localhost:8080/v1` with the `Qwen3-Omni-10k` model loaded. This model supports audio, vision, and language processing.

## Assets

All required assets (images, audio files) are stored in the `assets/` directory. These were downloaded from the original cookbook URLs to ensure fully local operation.

## Local Examples

The following examples demonstrate different multimodal capabilities using a local API endpoint. Each example has been run and documented:

### Audio Examples
1. [Audio Caption](audio_caption_local.py) - [Output](outputs/audio_caption_output.txt) - [Documentation](outputs/audio_caption_README.md)
   - Detailed audio descriptions

2. [Audio Function Call](audio_function_call_local.py) - [Output](outputs/audio_function_call_output.txt) - [Documentation](outputs/audio_function_call_README.md)
   - Audio-based function calling

3. [Audio Visual Dialogue](audio_visual_dialogue_local.py) - [Output](outputs/audio_visual_dialogue_output.txt) - [Documentation](outputs/audio_visual_dialogue_README.md)
   - Audio-video dialogue interactions

4. [Audio Visual Interaction](audio_visual_interaction_local.py) - [Output](outputs/audio_visual_interaction_output.txt) - [Documentation](outputs/audio_visual_interaction_README.md)
   - Audio-video interactions

5. [Audio Visual Question](audio_visual_question_local.py) - [Output](outputs/audio_visual_question_output.txt) - [Documentation](outputs/audio_visual_question_README.md)
   - Audio-video question answering

6. [Mixed Audio Analysis](mixed_audio_analysis_local.py) - [Output](outputs/mixed_audio_analysis_output.txt) - [Documentation](outputs/mixed_audio_analysis_README.md)
   - Analysis of mixed audio content

7. [Music Analysis](music_analysis_local.py) - [Output](outputs/music_analysis_output.txt) - [Documentation](outputs/music_analysis_README.md)
   - Music analysis and description

8. [Speech Recognition](speech_recognition_local.py) - [Output](outputs/speech_recognition_output.txt) - [Documentation](outputs/speech_recognition_README.md)
   - Speech-to-text conversion

9. [Speech Translation](speech_translation_local.py) - [Output](outputs/speech_translation_output.txt) - [Documentation](outputs/speech_translation_README.md)
   - Speech translation between languages

10. [Sound Analysis](sound_analysis_local.py) - [Output](outputs/sound_analysis_output.txt) - [Documentation](outputs/sound_analysis_README.md)
    - Sound identification and analysis

### Visual Examples
11. [Image Math](image_math_local.py) - [Output](outputs/image_math_output.txt) - [Documentation](outputs/image_math_README.md)
    - Mathematical problem solving with images

12. [Image Question](image_question_local.py) - [Output](outputs/image_question_output.txt) - [Documentation](outputs/image_question_README.md)
    - Image-based question answering

13. [Object Grounding](object_grounding_local.py) - [Output](outputs/object_grounding_output.txt) - [Documentation](outputs/object_grounding_README.md)
    - Object detection and localization

14. [OCR](ocr_local.py) - [Output](outputs/ocr_output.txt) - [Documentation](outputs/ocr_README.md)
    - Optical character recognition from images

### Other Examples
15. [Video Description](video_description_local.py) - [Output](outputs/video_description_output.txt) - [Documentation](outputs/video_description_README.md)
    - Video content description

## Usage

To run the examples:

```bash
# Make sure your local API server is running
python audio_caption_local.py
python image_question_local.py
```

## Creating Additional Local Examples

To create a local version of any other notebook file:

1. Convert the notebook to Python: `jupyter nbconvert --to python <notebook>.ipynb`
2. Replace the model loading and inference code with API calls to the local endpoint
3. Update any remote asset URLs to use local files in the `assets/` directory
4. Use the `run_model_local()` function pattern from existing examples

## Files Structure

```
.
├── audio_caption_local.py      # Audio captioning example
├── image_question_local.py     # Image question answering example
├── simple_test.py             # Simple API connectivity test
├── test_multimodal.py         # Multimodal functionality test
├── test_audio.py              # Audio functionality test
├── outputs/                   # Output files and documentation
│   ├── audio_caption_output.txt
│   ├── audio_caption_README.md
│   ├── ...
│   └── README.md              # Summary documentation
└── assets/                    # Local assets directory
    ├── caption1.mp3
    ├── caption2.mp3
    ├── caption3.mp3
    ├── 2621.jpg
    ├── 2233.jpg
    └── ... (all required assets)
```