import base64
import re
from io import BytesIO
from typing import List, Optional, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel
from embodied_eval.utils import Collator

try:
    from qwen_vl_utils import process_vision_info, vision_process
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils. Please install it via `pip install qwen-vl-utils`")

@register_model("pelican_vl")
class PelicanVL(BaseAPIModel):
    """
    Pelican-VL Model - 基于 Qwen2.5-VL 架构的具身智能大模型
    https://huggingface.co/collections/X-Humanoid/pelican-vl-10
    """

    def __init__(
            self,
            model_name_or_path: str = "X-Humanoid/Pelican1.0-VL-7B",
            device: Optional[str] = "cuda",
            device_map: Optional[str] = "cuda",
            max_length: Optional[int] = 2048,
            batch_size: Optional[Union[int, str]] = 1,
            max_new_tokens: Optional[int] = 1024,
            temperature: float = 0,
            do_sample: bool = False,
            top_p: Optional[int] = None,
            num_beams: Optional[int] = 1,
            use_cache: Optional[bool] = True,
            system_prompt: Optional[str] = "You are a helpful assistant.",
            use_flash_attention_2: Optional[bool] = False,
            max_num_frames: int = 32,
            fps: Optional[float] = None,
            min_pixels: int = 3126,
            max_pixels: int = 256*28*28,
            interleave_visuals: Optional[bool] = False,
            **kwargs,
    ) -> None:
        super().__init__()

        self.process_vision_info = process_vision_info
        if fps is not None:
            vision_process.FPS = fps

        # Handle distributed setup
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device
        
        eval_logger.info(f"Loading Pelican-VL model from {model_name_or_path}")
        if use_flash_attention_2:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map
            ).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        self._config = self._model.config
        self._max_length = max_length
        self.batch_size_per_gpu = int(batch_size)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.num_beams = num_beams
        self.use_cache = use_cache
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.interleave_visuals = interleave_visuals
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU], \
                "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
    
    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def max_length(self):
        return self._max_length

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def process_visuals(self, visual):
        """Process visuals for the model."""
        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
            processed_visual = {
                "type": "video",
                "video": visual,
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels
            }
        elif isinstance(visual, str) and visual.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            try:
                image = Image.open(visual).convert("RGB")
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                base64_bytes = base64.b64encode(buffer.getvalue())
                base64_string = base64_bytes.decode("utf-8")
                processed_visual = {
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{base64_string}",
                    "max_pixels": self.max_pixels,
                    "min_pixels": self.min_pixels
                }
            except Exception as e:
                eval_logger.warning(f"Failed to load image {visual}: {e}")
                processed_visual = None
        elif isinstance(visual, (Image.Image, np.ndarray)):
            if isinstance(visual, np.ndarray):
                visual = Image.fromarray(visual)
            base64_image = visual.convert("RGB")
            buffer = BytesIO()
            base64_image.save(buffer, format="JPEG")
            base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
            processed_visual = {
                "type": "image",
                "image": f"data:image/jpeg;base64,{base64_string}",
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels
            }
        else:
            eval_logger.warning(f"Unsupported visual type: {type(visual)}")
            processed_visual = None
        
        return processed_visual

    def build_messages(self, context: str, visuals: List[dict]) -> List[dict]:
        """Build messages for the model."""
        message = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        
        if not self.interleave_visuals:
            message.append({
                "role": "user",
                "content": visuals + [{"type": "text", "text": context}],
            })
        else:
            image_placeholders = re.findall(r"<image \d+>", context)
            content_parts = []
            text_parts = re.split(r"<image \d+>", context)
            if text_parts[0]:
                content_parts.append({"type": "text", "text": text_parts[0]})

            for i, placeholder in enumerate(image_placeholders):
                img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                image_idx = min(img_idx, len(visuals) - 1) if visuals else 0
                if visuals and image_idx < len(visuals):
                    content_parts.append(visuals[image_idx])
                if i + 1 < len(text_parts) and text_parts[i + 1]:
                    content_parts.append({"type": "text", "text": text_parts[i + 1]})

            message.append({
                "role": "user",
                "content": content_parts,
            })
        
        return message
    
    def respond(self, context, visuals, **gen_kwargs):
        """Generate a text response based on the given context and visual inputs."""
        if "<image>" in context:
            context = context.replace("<image>", "")

        processed_visuals = [self.process_visuals(visual) for visual in visuals]
        processed_visuals = [v for v in processed_visuals if v is not None]
        
        message = self.build_messages(context, processed_visuals)
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info([message])
        
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)
        
        gen_kwargs.setdefault("max_new_tokens", self.max_new_tokens)
        gen_kwargs.setdefault("do_sample", self.do_sample)
        gen_kwargs.setdefault("temperature", self.temperature)
        gen_kwargs.setdefault("top_p", self.top_p)
        gen_kwargs.setdefault("num_beams", self.num_beams)
        gen_kwargs.setdefault("use_cache", self.use_cache)

        cont = self.model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        text_output = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return text_output.strip()
