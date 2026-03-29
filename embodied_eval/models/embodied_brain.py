import base64
import decord
import re
import numpy as np
import torch

from io import BytesIO
from tqdm import tqdm
from PIL import Image

from accelerate import Accelerator, DistributedType
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from typing import List, Optional, Union
from loguru import logger as eval_logger

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel
from embodied_eval.utils import Collator

from qwen_vl_utils import process_vision_info

@register_model("embodied_brain")
class EmbodiedBrain(BaseAPIModel):
    """
    EmbodiedBrain Model (EmbodiedBrain-7B)
    Fine-tuned from Qwen2.5-VL for embodied AI tasks
    """

    def __init__(
            self,
            model_name_or_path: str = "/your/path/to/embodied-eval-main/embodied_eval/data/EmbodiedBrain-7B",
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
            use_flash_attention_2: Optional[bool] = True,
            max_num_frames: int = 32,
            min_pixels: int = 3126,
            max_pixels: int = 256*28*28,
            interleave_visuals: Optional[bool] = False,
            **kwargs,
    ) -> None:
        super().__init__()

        # Handle distributed setup
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device
        
        # Load model
        eval_logger.info(f"Loading EmbodiedBrain model from {model_name_or_path}")
        
        # Try to load with flash attention, fallback to standard attention if failed
        if use_flash_attention_2:
            try:
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device_map,
                    attn_implementation="flash_attention_2",
                ).eval()
                eval_logger.info("Successfully loaded model with Flash Attention 2")
            except (ImportError, Exception) as e:
                eval_logger.warning(f"Failed to load with Flash Attention 2: {e}")
                eval_logger.info("Falling back to standard attention implementation")
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name_or_path, 
                    torch_dtype=torch.bfloat16, 
                    device_map=self.device_map
                ).eval()
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name_or_path, 
                torch_dtype=torch.bfloat16, 
                device_map=self.device_map
            ).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, 
            max_pixels=max_pixels, 
            min_pixels=min_pixels
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Store configuration
        self._config = self._model.config
        self._max_length = max_length if getattr(self._config, "max_length", None) else self._config.max_length
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
        self.interleave_visuals = interleave_visuals

        # Set up distributed evaluation
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
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
        # return the associated transformers.AutoConfig for the given pretrained model.
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
            # Video file
            vr = decord.VideoReader(visual)
            first_frame = vr[0].asnumpy()
            height, width = first_frame.shape[:2]
            processed_visual = {
                "type": "video",
                "video": visual,
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels
            }
        elif isinstance(visual, str) and visual.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            # Image file path
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
        elif isinstance(visual, Image.Image) or isinstance(visual, np.ndarray):
            if isinstance(visual, np.ndarray):
                visual = Image.fromarray(visual)
            # Handle both single and multiple images
            base64_image = visual.convert("RGB")
            buffer = BytesIO()
            base64_image.save(buffer, format="JPEG")
            base64_bytes = base64.b64encode(buffer.getvalue())
            base64_string = base64_bytes.decode("utf-8")
            processed_visual = {
                "type": "image",
                "image": f"data:image/jpeg;base64,{base64_string}",
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels
            }
        else:
            # Unsupported visual type
            eval_logger.warning(f"Unsupported visual type: {type(visual)}")
            processed_visual = None

        return processed_visual

    def build_messages(self, context: str, visuals: List[dict]) -> List[dict]:
        """Build messages for the model."""
        # Ensure context is string
        if not isinstance(context, str):
            context = str(context)

        message = []
        if self.system_prompt:
            # Use simple string for system prompt to be more compatible
            message.append({"role": "system", "content": self.system_prompt})
        
        # Qwen2.5-VL expects content to be a list of dicts when visuals are present
        # Even if visuals is empty, using a list of dicts is safer for the processor
        content = []
        for v in visuals:
            content.append(v)
        content.append({"type": "text", "text": context})

        message.append(
            {
                "role": "user",
                "content": content,
            }
        )
        return message
    
    def respond(self, context, visuals, **gen_kwargs):
        """
        Generate a text response based on the given context and visual inputs.
        Args:
            context (str/list): The input text context for the response.
            visuals (list): A list of visual inputs (e.g., images or videos) to process.
            gen_kwargs (dict, optional): Additional keyword arguments for text generation.
        Returns:
            str: The generated text response.
        """   
        # 强制处理 context 为纯字符串，防止 vsibench 等任务传入列表导致拼接报错
        if isinstance(context, list):
            if len(context) > 0:
                context = context[0]
            else:
                context = ""
        context = str(context)

        # Process the request
        if "<image>" in context:
            context = context.replace("<image>", "")

        # Ensure visuals is a list
        if not isinstance(visuals, list):
            visuals = [visuals] if visuals is not None else []
        
        # Flatten nested lists if present
        flattened_visuals = []
        for visual in visuals:
            if isinstance(visual, list):
                flattened_visuals.extend(visual)
            else:
                flattened_visuals.append(visual)
        
        # Process visuals
        processed_visuals = []
        for visual in flattened_visuals:
            if visual is not None and visual != "":
                processed = self.process_visuals(visual)
                if processed is not None:
                    processed_visuals.append(processed)
        
        # Build the message
        try:
            message = self.build_messages(context, processed_visuals)
            
            # Use a robust chat template string to avoid jinja errors with multi-modal lists.
            # It also inserts vision tokens so image features align with tokens.
            robust_template = (
                "{% for message in messages %}"
                "{{ '<|im_start|>' + message['role'] + '\\n' }}"
                "{% if message['content'] is string %}"
                "{{ message['content'] }}"
                "{% else %}"
                "{% for item in message['content'] %}"
                "{% if item['type'] == 'text' %}{{ item['text'] }}"
                "{% elif item['type'] == 'image' %}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
                "{% elif item['type'] == 'video' %}{{ '<|vision_start|><|video_pad|><|vision_end|>' }}"
                "{% endif %}"
                "{% endfor %}"
                "{% endif %}"
                "{{ '<|im_end|>\\n' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
            )

            # Apply chat template using our robust template
            text = self.processor.apply_chat_template(
                message, 
                tokenize=False, 
                add_generation_prompt=True,
                chat_template=robust_template
            )
            image_inputs, video_inputs = process_vision_info([message])
        except Exception as e:
            eval_logger.error(f"Error building message or processing vision info: {e}")
            eval_logger.error(f"Context type: {type(context)}, Context: {context[:100] if isinstance(context, str) else context}")
            eval_logger.error(f"Processed visuals count: {len(processed_visuals)}")
            raise
        
        inputs = self.processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)

        # Get generation parameters
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = self.max_new_tokens
        if "do_sample" not in gen_kwargs:
            gen_kwargs["do_sample"] = self.do_sample
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = self.temperature
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = self.top_p
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = self.num_beams
        if "use_cache" not in gen_kwargs:
            gen_kwargs["use_cache"] = self.use_cache

        cont = self.model.generate(
            **inputs,
            **gen_kwargs
        )
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        text_output = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return text_output.strip()
