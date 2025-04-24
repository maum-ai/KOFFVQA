import os
import requests
from time import sleep
import random

from PIL import Image
import base64
from io import BytesIO

import torch
import torchvision.transforms as transforms
import transformers
from transformers import AutoProcessor, AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, MistralConfig, GenerationConfig
from huggingface_hub import repo_exists
from huggingface_hub.errors import HFValidationError

from .utils import *


class base_model(object):
    '''
    Base class for model compatibility implementation.
    '''
    def __init__(self, arc_class, modelpath:str, adapterpath:str='', gen_kwargs:dict={}, **kwargs) -> None:
        self.processor = AutoProcessor.from_pretrained(modelpath, trust_remote_code=True)
        if isinstance(arc_class, str):
            arc_class = getattr(transformers, arc_class)
        device = 'auto' if kwargs.get('auto_device', False) else 'cuda'
        model = arc_class.from_pretrained(modelpath, device_map=device, torch_dtype='auto', trust_remote_code=True)
        self.model = model.eval()
        if adapterpath!='':
            self.model.load_adapter(adapterpath)
        self.gen_kwargs = gen_kwargs
        self.kwargs = kwargs
        self.system_prompt = kwargs.get('sys_prompt', '')

    def generate(self, image:Image.Image, question:str) -> str:
        '''
        Method to generate the "prediction" value for one test case, given an
        image and a question in text form. Additions such as system prompts or
        modifications of the prompt may happen internally.
        '''
        messages = []
        if self.system_prompt != '':
            messages.append({'role': 'system', 'content': [
                {'type': 'text', 'text': self.system_prompt}
            ]})
        messages.append({'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': question}
        ]})
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(**inputs, **self.gen_kwargs)
        return self.processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


class qwen2d5omni(base_model):
    def __init__(self, arc_class, modelpath:str, adapterpath:str='', gen_kwargs:dict={}, **kwargs) -> None:
        from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
        self.processor = Qwen2_5OmniProcessor.from_pretrained(modelpath, trust_remote_code=True)
        # device = 'auto' if kwargs.get('auto_device', False) else 'cuda'
        device = 'cuda'
        model = Qwen2_5OmniModel.from_pretrained(modelpath, device_map=device, torch_dtype='auto', trust_remote_code=True, enable_audio_output=False)
        self.model = model.eval()
        if adapterpath!='':
            self.model.load_adapter(adapterpath)
        self.gen_kwargs = gen_kwargs
        self.kwargs = kwargs
        self.system_prompt = kwargs.get('sys_prompt', '')
class use_auto(base_model):
    def __init__(self, arc_class, modelpath, adapterpath = '', gen_kwargs = {}, **kwargs):
        super().__init__(AutoModelForCausalLM, modelpath, adapterpath, gen_kwargs, **kwargs)
class use_automodel_and_tokenizer(base_model):
    def __init__(self, arc_class, modelpath: str, adapterpath: str = '', gen_kwargs: dict = {}, **kwargs) -> None:
        super().__init__(AutoModel, modelpath, adapterpath, gen_kwargs, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath, trust_remote_code=True)


class minicpmv(use_automodel_and_tokenizer):
    def generate(self, image: Image.Image, question: str) -> str:
        msgs = [{'role': 'user', 'content': [image, question]}]
        return self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)

class internvl(use_automodel_and_tokenizer):
    def generate(self, image: Image.Image, question: str) -> str:
        pixel_values = internvl_load_image(image).to(self.model.dtype).cuda()
        return self.model.chat(self.tokenizer, pixel_values, '<image>\n'+question, self.gen_kwargs)

class ovis(base_model):
    def __init__(self, arc_class, modelpath: str, adapterpath: str = '', gen_kwargs: dict = {}, **kwargs) -> None:
        device = 'auto' if kwargs.get('auto_device', False) else 'cuda'
        model = AutoModelForCausalLM.from_pretrained(modelpath, device_map=device, torch_dtype='auto', trust_remote_code=True)
        self.model = model.eval()
        if adapterpath!='':
            self.model.load_adapter(adapterpath)
        self.gen_kwargs = gen_kwargs
        self.kwargs = kwargs
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.gen_kwargs.update(dict(
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.text_tokenizer.pad_token_id,
        ))
    def generate(self, image: Image.Image, question: str) -> str:
        query = '<image>\n' + question
        _, input_ids, pixel_values = self.model.preprocess_inputs(query, [image])
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)]
        output = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **self.gen_kwargs)
        return self.text_tokenizer.decode(output[0], skip_special_tokens=True)

class internlm_xcomposer_2(use_automodel_and_tokenizer):
    def __init__(self, arc_class, modelpath: str, adapterpath: str = '', gen_kwargs: dict = {}, **kwargs) -> None:
        super().__init__(arc_class, modelpath, adapterpath, gen_kwargs, **kwargs)
        self.model.tokenizer = self.tokenizer
        self.vis_processor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
    def generate(self, image:Image.Image, question:str) -> str:
        image = xcomposer_Image_transform(image)
        image = self.vis_processor(image).unsqueeze(0).to(self.model.dtype).cuda()
        response, _ = self.model.chat(self.processor, question, [image], **self.gen_kwargs)
        return response

class llava_next(base_model):
    def generate(self, image:Image.Image, question:str) -> str:
        messages = []
        if self.system_prompt != '': messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': f'<image>\n{question}'})
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, image, return_tensors='pt').to(self.model.dtype).to('cuda')
        output = self.model.generate(**inputs, **self.gen_kwargs)
        return self.processor.tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

class llava_auto(base_model):
    def __init__(self, arc_class:str, modelpath: str, adapterpath: str = '', gen_kwargs: dict = {}, **kwargs) -> None:
        from llava.model.language_model import llava_llama, llava_qwen
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX

        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        class_map = {
            'LlavaLlamaForCausalLM': llava_llama.LlavaLlamaForCausalLM,
            'LlavaQwenForCausalLM': llava_qwen.LlavaQwenForCausalLM,
        }
        assert arc_class in class_map, 'This LLaVA architecture is not supported.'
        device = 'auto' if kwargs.get('auto_device', False) else 'cuda'
        self.model = class_map[arc_class].from_pretrained(modelpath, device_map=device, torch_dtype='auto', trust_remote_code=True)
        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(self.model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device)
        if device != 'auto':
            vision_tower.to(device=device, dtype=self.model.dtype)
        self.image_processor = vision_tower.image_processor
        self.model.eval()
        if adapterpath!='':
            self.model.load_adapter(adapterpath)
        self.gen_kwargs = gen_kwargs
        self.kwargs = kwargs
        self.system_prompt = kwargs.get('sys_prompt', '')

    def generate(self, image: Image.Image, question: str) -> str:
        image_tensor = self.process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=self.model.dtype, device='cuda') for _image in image_tensor]
        try:
            messages = []
            if self.system_prompt != '': messages.append({'role': 'system', 'content': [{'type': 'text', 'text': self.system_prompt}]})
            messages.append({'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]})
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            messages = []
            if self.system_prompt != '': messages.append({'role': 'system', 'content': self.system_prompt})
            messages.append({'role': 'user', 'content': f'<image>\n{question}'})
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        output = self.model.generate(input_ids, images=image_tensor, image_sizes=[image.size], **self.gen_kwargs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

class molmo(base_model):
    def __init__(self, arc_class, modelpath: str, adapterpath: str = '', gen_kwargs: dict = {}, **kwargs) -> None:
        self.processor = AutoProcessor.from_pretrained(modelpath, trust_remote_code=True)
        device = 'auto' if kwargs.get('auto_device', False) else 'cuda'
        model = AutoModelForCausalLM.from_pretrained(modelpath, device_map=device, torch_dtype='auto', trust_remote_code=True)
        self.model = model.eval()
        if adapterpath!='':
            self.model.load_adapter(adapterpath)
        self.gen_kwargs = gen_kwargs
        self.kwargs = kwargs
        self.system_prompt = kwargs.get('sys_prompt', '')
    def generate(self, image: Image.Image, question: str) -> str:
        inputs = self.processor.process(images=[image], text=question)
        inputs = {k: v.to('cuda').unsqueeze(0) for k, v in inputs.items()}
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(stop_strings="<|endoftext|>", **self.gen_kwargs),
            tokenizer=self.processor.tokenizer
        )
        return self.processor.tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


class pixtral(base_model):
    def __init__(self, arc_class, modelpath: str, adapterpath: str = '', gen_kwargs: dict = {}, **kwargs) -> None:
        assert os.path.exists(modelpath), 'Mistral models require pre-downloaded local path.'

        from mistral_inference.transformer import Transformer
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from mistral_inference.generate import generate
        from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageChunk
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        self.mistral_generate = generate
        self.UserMessage = UserMessage
        self.TextChunk = TextChunk
        self.ImageChunk = ImageChunk
        self.ChatCompletionRequest = ChatCompletionRequest

        self.tokenizer = MistralTokenizer.from_file(os.path.join(modelpath, 'tekken.json'))
        self.model = Transformer.from_folder(modelpath)

        self.gen_kwargs = gen_kwargs
        self.kwargs = kwargs
        self.system_prompt = kwargs.get('sys_prompt', '')

    def generate(self, image: Image.Image, question: str) -> str:
        completion_request = self.ChatCompletionRequest(messages=[
            self.UserMessage(content=[self.ImageChunk(image=image),
                                      self.TextChunk(text=question)])
        ])
        encoded = self.tokenizer.encode_chat_completion(completion_request)
        out_tokens, _ = self.mistral_generate([encoded.tokens], self.model, images=[encoded.images], eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id, **self.gen_kwargs)
        return self.tokenizer.decode(out_tokens[0])

class pixtral_hf(base_model):
    def __init__(self, arc_class, modelpath: str, adapterpath: str = '', gen_kwargs: dict = {}, **kwargs) -> None:
        super().__init__(arc_class, modelpath, adapterpath, gen_kwargs, **kwargs)
        assert hasattr(self.model.config, 'vision_config') and hasattr(self.model.config.vision_config, 'model_type') and self.model.config.vision_config.model_type, "Not a Pixtral model."
    def generate(self, image:Image.Image, question:str) -> str:
        prompt = f'<s>[INST][IMG]\n{question}[/INST]'
        inputs = self.processor(text=prompt, images=[image], return_tensors='pt').to('cuda')
        output = self.model.generate(**inputs, **self.gen_kwargs)
        return self.processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

class aria(base_model):
    def generate(self, image:Image.Image, question:str) -> str:
        messages = []
        if self.system_prompt != '':
            messages.append({'role': 'system', 'content': [
                {'type': 'text', 'text': self.system_prompt}
            ]})
        messages.append({'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': question}
        ]})
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to('cuda')
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        output = self.model.generate(**inputs, stop_strings=["<|im_end|>"], tokenizer=self.processor.tokenizer, **self.gen_kwargs)
        return self.processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).replace('<|im_end|>', '')

class phi3v(base_model):
    def __init__(self, arc_class, modelpath, adapterpath = '', gen_kwargs = {}, **kwargs):
        self.processor = AutoProcessor.from_pretrained(modelpath, trust_remote_code=True, num_crops=16)
        device = 'auto' if kwargs.get('auto_device', False) else 'cuda'
        model = AutoModelForCausalLM.from_pretrained(modelpath, device_map=device, torch_dtype='auto', trust_remote_code=True)
        self.model = model.eval()
        if adapterpath!='':
            self.model.load_adapter(adapterpath)
        self.gen_kwargs = gen_kwargs
        self.kwargs = kwargs
        self.system_prompt = kwargs.get('sys_prompt', '')
    def generate(self, image, question):
        messages = []
        if self.system_prompt != '':
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': f'<|image_1|>\n{question}'})
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **self.gen_kwargs)
        return self.processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

class phi4m(base_model):
    def __init__(self, arc_class, modelpath, adapterpath = '', gen_kwargs = {}, **kwargs):
        self.processor = AutoProcessor.from_pretrained(modelpath, trust_remote_code=True, num_crops=16)
        device = 'auto' if kwargs.get('auto_device', False) else 'cuda'
        model = AutoModelForCausalLM.from_pretrained(modelpath, device_map=device, torch_dtype='auto', trust_remote_code=True)
        self.model = model.eval()
        if adapterpath!='':
            self.model.load_adapter(adapterpath)
        self.gen_kwargs = gen_kwargs
        self.kwargs = kwargs
        self.generation_config = GenerationConfig.from_pretrained(modelpath)
    def generate(self, image, question):
        prompt = f'<|user|><|image_1|>{question}<|end|><|assistant|>'
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(**inputs, generation_config=self.generation_config, **self.gen_kwargs)
        return self.processor.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

class hyperclovax(base_model):
    def __init__(self, arc_class, modelpath: str, adapterpath: str = '', gen_kwargs: dict = {}, **kwargs) -> None:
        self.processor = AutoProcessor.from_pretrained(modelpath, trust_remote_code=True)
        device = 'cuda'
        model = AutoModelForCausalLM.from_pretrained(modelpath, device_map=device, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.model = model.eval()
        if adapterpath!='':
            self.model.load_adapter(adapterpath)
        self.gen_kwargs = gen_kwargs
        self.kwargs = kwargs
        self.system_prompt = kwargs.get('sys_prompt', '')
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath, trust_remote_code=True)
    def generate(self, image: Image.Image, question: str) -> str:
        messages = []

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_base64 = f"data:image/jpeg;base64,{img_base64}"

        messages.append({'role': 'user', 'content': {'type': 'image', 'image': img_base64}})
        messages.append({'role': 'user', 'content': {'type': 'text', 'text': question}})

        new_vlm_chat, all_images, is_video_list = self.processor.load_images_videos(messages)
        preprocessed = self.processor(all_images, is_video_list=is_video_list).to(dtype=self.model.dtype)
        input_ids = self.tokenizer.apply_chat_template(new_vlm_chat, return_tensors="pt", tokenize=True, add_generation_prompt=True)
        input_ids = input_ids.to(self.model.device)

        output = self.model.generate(input_ids=input_ids, **self.gen_kwargs, **preprocessed)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

class api_model(base_model):
    def __init__(self, arc_class, modelpath: str, adapterpath: str = '', gen_kwargs: dict = {}, **kwargs) -> None:
        self.modelname = modelpath
        self.gen_kwargs = gen_kwargs
        self.kwargs = kwargs
        self.api_key = os.environ['API_KEY']
    def try_generate(self, image: Image.Image, question: str) -> str:
        '''
        Generation sub-method for API models. Tries to generate a response, but
        return empty string upon failure.
        '''
        raise NotImplementedError()
    def generate(self, image: Image.Image, question: str) -> str:
        '''
        Generation method for API models. Tries generating a response at max 10
        times with randomized exponential backoff, and raises an exception if
        it fails all 10 times.
        '''
        delay = 1.
        gen = self.try_generate(image, question)
        if gen != '': return gen
        for _ in range(9):
            print(f'API failed to generate. Waiting for {delay:.1f} seconds before retrying.')
            sleep(delay)
            delay *= random.uniform(1.5, 2)
            gen = self.try_generate(image, question)
            if gen != '': return gen
        print('Generation failed after 10 tries. Aborting.')
        raise RuntimeError('API generation failed')


class openai_api_model(api_model):
    def try_generate(self, image: Image.Image, question: str) -> str:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        payload = {
            'model': self.modelname,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{img_encode(image)}"}},
                    {'type': 'text', 'text': question}
                ]
            }],
        }
        payload.update(self.gen_kwargs)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if not response.ok: return ''
        try: return response.json()['choices'][0]['message']['content']
        except (KeyError, IndexError): return ''


class gemini_api_model(api_model):
    def try_generate(self, image: Image.Image, question: str) -> str:
        header = {
            'Content-Type': 'application/json',
        }
        payload = {'contents': [{
            'parts': [
                {'inlineData': {
                    'mimeType': 'image/jpeg',
                    'data': img_encode(image)
                }},
                {'text': question},
            ], 'role': 'user'
        }], 'generationConfig': self.gen_kwargs}
        response = requests.post(f'https://generativelanguage.googleapis.com/v1beta/models/{self.modelname}:generateContent?key={self.api_key}', headers=header, json=payload)
        if not response.ok: return ''
        try: return response.json()['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError): return ''


class claude_api_model(api_model):
    def try_generate(self, image: Image.Image, question: str) -> str:
        header = {
            'content-type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        }
        payload = {
            'model': self.modelname,
            'messages': [
                {'role': 'user', 'content': [
                    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': img_encode(image)}},
                    {'type': 'text', 'text': question}
                ]}
            ],
        }
        payload.update(self.gen_kwargs)
        response = requests.post('https://api.anthropic.com/v1/messages', headers=header, json=payload)
        if not response.ok: return ''
        try: return response.json()['content'][0]['text']
        except (KeyError, IndexError): return ''


def auto_model(modelpath:str, adapterpath:str='', **kwargs) -> base_model:
    '''
    Automatically creates a model object based on the given paths.
    '''
    max_tokens = 512

    api_supported = {
        'gemini-2.5-pro': (gemini_api_model, dict(maxOutputTokens=12000)),
        'gemini': (gemini_api_model, dict(maxOutputTokens=max_tokens)),
        'claude': (claude_api_model, dict(max_tokens=max_tokens)),
        'gpt-4o': (openai_api_model, dict(max_completion_tokens=max_tokens)),
        'gpt-4': (openai_api_model, dict(max_completion_tokens=max_tokens)),
        'o1': (openai_api_model, dict(max_completion_tokens=12000)),
        'o3': (openai_api_model, dict(max_completion_tokens=12000)),
        'o4-mini': (openai_api_model, dict(max_completion_tokens=12000)),
    }
    for modelpattern in api_supported:
        if modelpattern in modelpath:
            base_cls, gen_kw = api_supported[modelpattern]
            return base_cls(None, modelpath, '', gen_kw, **kwargs)

    if not os.path.exists(modelpath):
        try:
            if not repo_exists(modelpath):
                raise ValueError('Model is either unsupported, or nonexistent.')
        except HFValidationError:
            raise ValueError('Model is either unsupported, or nonexistent.')

    try: import llava
    except ImportError: pass # fix for model registry issue

    base_supported = {
        'MllamaForConditionalGeneration': (base_model, dict(max_new_tokens=max_tokens, do_sample=True, temperature=0.6, top_p=0.9)),
        'Qwen2VLForConditionalGeneration': (base_model, dict(max_new_tokens=max_tokens, do_sample=False)),
        'MiniCPMV': (minicpmv, dict(max_new_tokens=max_tokens, do_sample=True)),
        'InternVLChatModel': (internvl, dict(max_new_tokens=max_tokens, do_sample=True)),
        'Ovis': (ovis, dict(max_new_tokens=max_tokens, do_sample=False)),
        'InternLMXComposer2ForCausalLM': (internlm_xcomposer_2, dict(max_new_tokens=max_tokens, do_sample=False, use_meta=True)),
        'LlavaNextForConditionalGeneration': (llava_next, dict(max_new_tokens=max_tokens, do_sample=False)),
        'LlavaLlamaForCausalLM': (llava_auto, dict(max_new_tokens=max_tokens, do_sample=False)),
        'LlavaQwenForCausalLM': (llava_auto, dict(max_new_tokens=max_tokens, do_sample=False)),
        'LlavaForConditionalGeneration': (pixtral_hf, dict(max_new_tokens=max_tokens, do_sample=True)),
        'MolmoForCausalLM': (molmo, dict(max_new_tokens=max_tokens, do_sample=True)),
        'Idefics3ForConditionalGeneration': (base_model, dict(max_new_tokens=max_tokens, do_sample=False)),
        'AriaForConditionalGeneration': (aria, dict(max_new_tokens=max_tokens, do_sample=True, temperature=0.9)),
        'Phi3VForCausalLM': (phi3v, dict(max_new_tokens=max_tokens, do_sample=False)),
        'Qwen2_5OmniModel': (qwen2d5omni, dict(max_new_tokens=max_tokens, do_sample=False, return_audio=False)),
        'AyaVisionForConditionalGeneration': (base_model, dict(max_new_tokens=max_tokens, do_sample=True, temperature=0.3)),
        'Gemma3ForConditionalGeneration': (base_model, dict(max_new_tokens=max_tokens, do_sample=False)),
        'Phi4MMForCausalLM': (phi4m, dict(max_new_tokens=max_tokens, do_sample=False)),
        'Llama4ForConditionalGeneration': (base_model, dict(max_new_tokens=max_tokens)),
        'Mistral3ForConditionalGeneration': (base_model, dict(max_new_tokens=max_tokens, do_sample=True, temperature=0.15)),
        'KimiVLForConditionalGeneration': (use_auto, dict(max_new_tokens=max_tokens, do_sample=True, temperature=0.2)),
        'Llama4ForConditionalGeneration': (base_model, dict(max_new_tokens=max_tokens)),
        'HCXVisionForCausalLM': (hyperclovax, dict(max_length=8192, do_sample=True, top_p=0.6, temperature=0.5, repetition_penalty=1.0)),
    }
    cfg = AutoConfig.from_pretrained(modelpath, trust_remote_code=True)
    # exception case for pixtral model, because of the different format
    if isinstance(cfg, MistralConfig):
        return pixtral(None, modelpath, '', dict(max_tokens=max_tokens, temperature=0.35), **kwargs)
    elif hasattr(cfg, 'language_config') and hasattr(cfg.language_config, 'architectures') and cfg.language_config.architectures[0] in base_supported:
        arc_cls = cfg.language_config.architectures[0]
        base_cls, gen_kw = base_supported[arc_cls]
        return base_cls(arc_cls, modelpath, adapterpath, gen_kw, **kwargs)
    elif hasattr(cfg, 'architectures') and cfg.architectures[0] in base_supported:
        arc_cls = cfg.architectures[0]
        base_cls, gen_kw = base_supported[arc_cls]
        return base_cls(arc_cls, modelpath, adapterpath, gen_kw, **kwargs)

    raise NotImplementedError('Model architecture not found or not yet supported.')


