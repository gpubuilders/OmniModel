#!/usr/bin/env python
# coding: utf-8

# ### Image Math with Qwen3-Omni

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
                                            thinker_do_sample=False,
                                            speaker="Ethan", 
                                            use_audio_in_video=use_audio_in_video,
                                            return_audio=return_audio)
        response = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if audio is not None:
            audio = np.array(audio.reshape(-1).detach().cpu().numpy() * 32767).astype(np.int16)
        return response, audio
    else:
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=1e-2, top_p=0.1, top_k=1, max_tokens=8192)
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

from IPython.display import Video
from IPython.display import Image
from IPython.display import Audio

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

USE_TRANSFORMERS = False
TRANSFORMERS_USE_FLASH_ATTN2 = True

model, processor = _load_model_processor()

USE_AUDIO_IN_VIDEO = True
RETURN_AUDIO = False


# In[ ]:


image_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/5195.jpg"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "The 3-arm lawn sprinkler of Fig. P3.114 receives 20°C water through the center at 2.7 m^3/hr. If collar friction is neglected, what is the steady rotation rate in rev/min for $\\theta $ = 40°?"},
            {"type": "image", "image": image_path},
            {"type": "text", "text": "A, 317 rev/min, B, 414 rev/min, C, 400 rev/min, D, NaN, choose one of the options."""} 
        ]
    }
]

display(Image(image_path, width=640, height=360))

response, _ = run_model(model=model, messages=messages, processor=processor, return_audio=RETURN_AUDIO, use_audio_in_video=USE_AUDIO_IN_VIDEO)

print(response)


# In[ ]:


image_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/4181.jpg"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Suppose we have a training set with m=3 examples,plotted below.Our hypothes is representation is $h_{\\theta}(x) = \\theta_1x$, with parameter $\\theta_1$. The cost function $J(\\theta_1)$ is $J(\\theta_1)$=${1\\over2m}\\sum\\nolimits_{i=1}^m(h_\\theta(x^i) - y^i)^2$. What is $J(0)$?"},
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Options:\nA. 0\nB. 1/6\nC. 1\nD. 14/6\nPlease select the correct answer from the options above."}
        ]
    }
]

display(Image(image_path, width=640, height=360))

response, _ = run_model(model=model, messages=messages, processor=processor, return_audio=RETURN_AUDIO, use_audio_in_video=USE_AUDIO_IN_VIDEO)

print(response)

