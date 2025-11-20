#!/usr/bin/env python
# coding: utf-8

# ### Qwen3-Omni-30B-A3B-Captioner

# Qwen3-Omni-30B-A3B-Captioner is a powerful fine-grained audio analysis model, built upon the Qwen3-Omni-30B-A3B-Instruct base model. It is specifically designed to generate accurate and comprehensive content descriptions in complex and diverse audio scenarios. Without requiring any additional prompting, the model can automatically parse and describe various types of audio content, ranging from complex speech and environmental sounds to music and cinematic sound effects, delivering stable and reliable outputs even in multi-source, mixed audio environments.
# 
# In terms of speech understanding, Qwen3-Omni-30B-A3B-Captioner excels at identifying multiple speaker emotions, multilingual expressions, and layered intentions. It can also perceive cultural context and implicit information within the audio, enabling a deep comprehension of the underlying meaning behind the spoken words. In non-speech scenarios, the model demonstrates exceptional sound recognition and analysis capabilities, accurately distinguishing and describing intricate layers of real-world sounds, ambient atmospheres, and dynamic audio details in film and media.

# **Note**: Qwen3-Omni-30B-A3B-Captioner is a single-turn model that accepts only one audio input per inference. It does not accept any text prompts and supports **audio input only**, with **text output only**. As Qwen3-Omni-30B-A3B-Captioner is designed for generating fine‑grained descriptions of audio, excessively long audio clips may diminish detail perception. We recommend, as a best practice, limiting audio length to no more than 30 seconds.

# In[1]:


import os
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import warnings
import numpy as np

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor

def _load_model_processor():
    if USE_TRANSFORMERS:
        from transformers import Qwen3OmniMoeForConditionalGeneration
        if TRANSFORMERS_USE_FLASH_ATTN2:
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL_PATH,
                                                                         dtype='auto',
                                                                         attn_implementation='flash_attention_2',
                                                                         device_map="auto")
        else:
            model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL_PATH, device_map="auto", dtype='auto')
    else:
        from vllm import LLM
        model = LLM(
            model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={'image': 1, 'video': 3, 'audio': 3},
            max_num_seqs=1,
            max_model_len=32768,
            seed=1234,
        )

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    return model, processor

def run_model(model, processor, messages, return_audio, use_audio_in_video):
    if USE_TRANSFORMERS:
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=use_audio_in_video)
        inputs = inputs.to(model.device).to(model.dtype)
        text_ids, audio = model.generate(**inputs, 
                                            thinker_return_dict_in_generate=True,
                                            thinker_max_new_tokens=8192, 
                                            thinker_do_sample=True,
                                            thinker_top_p=0.95,
                                            thinker_top_k=20,
                                            thinker_temperature=0.6,
                                            speaker="Chelsie", 
                                            use_audio_in_video=use_audio_in_video,
                                            return_audio=return_audio)
        response = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if audio is not None:
            audio = np.array(audio.reshape(-1).detach().cpu().numpy() * 32767).astype(np.int16)
        return response, audio
    else:
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=8192)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = {'prompt': text, 'multi_modal_data': {}, "mm_processor_kwargs": {"use_audio_in_video": use_audio_in_video}}
        if images is not None: inputs['multi_modal_data']['image'] = images
        if videos is not None: inputs['multi_modal_data']['video'] = videos
        if audios is not None: inputs['multi_modal_data']['audio'] = audios
        outputs = model.generate(inputs, sampling_params=sampling_params)
        response = outputs[0].outputs[0].text
        return response, None



# In[2]:


import librosa
import audioread

from IPython.display import Audio

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Captioner"

USE_TRANSFORMERS = False
TRANSFORMERS_USE_FLASH_ATTN2 = True

model, processor = _load_model_processor()

USE_AUDIO_IN_VIDEO = True


# In[6]:


audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/captioner-case1.wav"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path}
        ]
    }
]

display(Audio(librosa.load(audioread.ffdec.FFmpegAudioFile(audio_path), sr=16000)[0], rate=16000))

response, _ = run_model(model=model, messages=messages, processor=processor, return_audio=False, use_audio_in_video=USE_AUDIO_IN_VIDEO)

print(response)


# In[4]:


audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/captioner-case2.wav"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path}
        ]
    }
]

display(Audio(librosa.load(audioread.ffdec.FFmpegAudioFile(audio_path), sr=16000)[0], rate=16000))

response, _ = run_model(model=model, messages=messages, processor=processor, return_audio=False, use_audio_in_video=USE_AUDIO_IN_VIDEO)

print(response)


# In[5]:


audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/captioner-case3.wav"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path}
        ]
    }
]

display(Audio(librosa.load(audioread.ffdec.FFmpegAudioFile(audio_path), sr=16000)[0], rate=16000))

response, _ = run_model(model=model, messages=messages, processor=processor, return_audio=False, use_audio_in_video=USE_AUDIO_IN_VIDEO)

print(response)

