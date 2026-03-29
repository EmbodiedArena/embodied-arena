import torch
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from typing import List, Optional, Union

import sys
import os
cambrian_parent_path = "/your/path/to/embodied-eval-main/embodied_eval/data/Cambrian-S-7B/cambrian-s"
if cambrian_parent_path not in sys.path:
    sys.path.append(cambrian_parent_path)

from cambrian.model.builder import load_pretrained_model
from cambrian.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from cambrian.conversation import conv_templates

from cambrian.constants import IMAGE_TOKEN_INDEX
from cambrian.conversation import conv_templates
from cambrian.model.builder import load_pretrained_model
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, expand2square
from decord import VideoReader, cpu
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel

@register_model("cambrian")
class Cambrian(BaseAPIModel):
    """
    适配 Cambrian-S-7B 
    Model Path: "nyu-visionx/Cambrian-S-7B"
    需要修改conda环境为 cambrian！！
    """

    def __init__(
            self,
            model_name_or_path: str = "nyu-visionx/Cambrian-S-7B",
            device: Optional[str] = "cuda",
            device_map: Optional[str] = "cuda",
            conv_mode: str = "qwen_2", # Cambrian-S 使用 qwen_2 模板
            max_new_tokens: Optional[int] = 1024,
            temperature: float = 0,
            do_sample: bool = False,
            **kwargs,
    ) -> None:
        super().__init__()

        # 1. 自动处理分布式设备映射
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
            self.device_map = {"": accelerator.process_index}
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.device_map = device_map
            self._device = torch.device(device)
            self._rank = 0
            self._world_size = 1

        # 2. 加载 Cambrian 模型
        eval_logger.info(f"Loading Cambrian model from {model_name_or_path}")
        model_name = get_model_name_from_path(model_name_or_path)
        
        # 加载逻辑
        self._tokenizer, self._model, self._image_processor, _ = load_pretrained_model(
            model_path=model_name_or_path,
            model_base=None,
            model_name=model_name,
            device_map=self.device_map
        )
        self._model.eval()

        # 3. 配置参数
        self.conv_mode = conv_mode
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_num_frames = int(kwargs.get("max_num_frames", 8))
        
        # 获取图像处理相关的配置
        if hasattr(self._model, 'get_vision_tower'):
            self.image_token_len = self._model.get_vision_tower().num_patches
        elif hasattr(self._model, 'model') and hasattr(self._model.model, 'get_vision_tower'):
            # 对于某些封装结构，vision_tower 在底层的 .model 属性中
            self.image_token_len = self._model.model.get_vision_tower().num_patches
        else:
            eval_logger.warning("Could not find get_vision_tower method, using default value.")
            self.image_token_len = 576

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

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
        """将输入转为 PIL Image"""
        if isinstance(visual, str):
            # 处理路径或 base64 (简化处理，直接用 PIL 打开路径)
            return Image.open(visual).convert("RGB")
        elif isinstance(visual, np.ndarray):
            return Image.fromarray(visual)
        elif isinstance(visual, Image.Image):
            return visual.convert("RGB")
        return None

    def process_video_with_decord(self,video_file, model_cfg, num_threads=-1):

        if num_threads < 1:
            vr = VideoReader(video_file, ctx=cpu(0))
        else:
            vr = VideoReader(video_file, ctx=cpu(0), num_threads=num_threads)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        avg_fps = round(vr.get_avg_fps() / model_cfg.video_fps)
        frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
        frame_time = [i / avg_fps for i in frame_idx]

        if model_cfg.video_max_frames > 0:
            if len(frame_idx) > model_cfg.video_max_frames or model_cfg.video_force_sample:
                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, model_cfg.video_max_frames, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
                frame_time = [i / vr.get_avg_fps() for i in frame_idx]

        video = vr.get_batch(frame_idx).asnumpy()
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

        num_frames_to_sample = num_frames = len(frame_idx)
        vr.seek(0)
        return video, video_time, frame_time, num_frames_to_sample


    def process_videos(self,videos, image_processor, model_cfg, num_threads=-1):

        processor_aux_list = image_processor

        new_videos_aux_list = []
        video_sizes = []

        for video_path in videos:
            video, video_time, frame_time, num_frames_to_sample = self.process_video_with_decord(video_path, model_cfg, num_threads=num_threads)
            video_sizes.append((video.shape[2], video.shape[1], video.shape[0]))  # W, H, T
            video_pil = [Image.fromarray(video[_], mode="RGB") for _ in range(video.shape[0])]  # covert to PIL.Image.Image

            video_aux_list = []
            for processor_aux in processor_aux_list:
                video_aux = video_pil
                video_aux = [expand2square(image, tuple(int(x * 255) for x in processor_aux.image_mean)) for image in video_aux]
                # preprocess 返回通常是 [N, C, H, W]
                t = processor_aux.preprocess(video_aux, return_tensors="pt")["pixel_values"]
                if t.ndim == 3: t = t.unsqueeze(0) # 确保是 [N, C, H, W]
                video_aux_list.append(t)

            new_videos_aux_list.append(video_aux_list)

        new_videos_aux_list = [list(batch_video_aux) for batch_video_aux in zip(*new_videos_aux_list)]
        
        # 鲁棒性合并：确保每个 Tower 的视频帧 Batch 维度对齐
        final_videos = []
        for tower_tensors in new_videos_aux_list:
            # tower_tensors: list of [N, C, H, W]
            try:
                combined = torch.stack(tower_tensors) # [B, N, C, H, W]
            except RuntimeError:
                # 维度不一致时（如帧数不同），对齐到最大维度
                max_n = max(t.shape[0] for t in tower_tensors)
                aligned = []
                for t in tower_tensors:
                    if t.shape[0] < max_n:
                        pad = torch.zeros((max_n - t.shape[0], *t.shape[1:]), device=t.device, dtype=t.dtype)
                        t = torch.cat([t, pad], dim=0)
                    aligned.append(t)
                combined = torch.stack(aligned)
            final_videos.append(combined)

        return final_videos, video_sizes, (video_time, frame_time, num_frames_to_sample)

    def respond(self, context, visuals, **gen_kwargs):
        # 1. 分类输入
        video_paths = []
        image_objs = []
        for v in visuals:
            if v is None: continue
            if isinstance(v, str) and v.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_paths.append(v)
            else:
                img = self.process_visuals(v)
                if img: image_objs.append(img)

        # 2. 注入官方所需配置
        self._model.config.video_max_frames = gen_kwargs.get("video_max_frames", self.max_num_frames)
        self._model.config.video_fps = gen_kwargs.get("video_fps", 1)
        self._model.config.video_force_sample = True
        self._model.config.image_aspect_ratio = "anyres"
        self._model.config.anyres_max_subimages = 9

        # 3. 预处理
        if video_paths:
            # 视频文件按原逻辑处理
            visual_tensors, visual_sizes, _ = self.process_videos(
                video_paths, self._image_processor, self._model.config
            )
            num_image_tokens = 1 # 视频通常占一个占位符
        elif image_objs:
            # 对于 UniEQA 的多图输入，将它们拼接成一张大图。
            # 这是处理多视角/多帧数据最稳妥的方式，能绕过 LLaVA 架构对多 <image> token 的支持限制
            if len(image_objs) > 1:
                num_imgs = len(image_objs)
                grid_size = int(np.ceil(np.sqrt(num_imgs)))
                w, h = image_objs[0].size
                combined_img = Image.new('RGB', (w * grid_size, h * grid_size), (0, 0, 0))
                for i, img in enumerate(image_objs):
                    combined_img.paste(img, ((i % grid_size) * w, (i // grid_size) * h))
                image_objs = [combined_img]
            
            # 使用单图模式处理
            visual_tensors, visual_sizes = process_images(image_objs, self._image_processor, self._model.config)
            num_image_tokens = 1
        else:
            return "Error: No visual input"

        # 4. Prompt 与 Tokenize
        clean_context = context.replace(DEFAULT_IMAGE_TOKEN, "").strip()
        query = (DEFAULT_IMAGE_TOKEN + "\n") * num_image_tokens + clean_context
        
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)

        # 5. 生成
        with torch.inference_mode():
            if visual_tensors is not None:
                # 统一转换为 bfloat16 并移到 device
                visual_tensors = [t.to(self.device, dtype=torch.bfloat16) for t in visual_tensors]

            if not isinstance(visual_sizes, list) and visual_sizes is not None:
                visual_sizes = [visual_sizes]

            if visual_tensors is not None and len(visual_tensors) == 0:
                visual_tensors = None
                visual_sizes = None

            output_ids = self._model.generate(
                inputs=input_ids,
                images=visual_tensors,
                image_sizes=visual_sizes,
                use_cache=True,
                do_sample=True if gen_kwargs.get("temperature", self.temperature) > 0 else False,
                temperature=gen_kwargs.get("temperature", self.temperature),
                max_new_tokens=gen_kwargs.get("max_new_tokens", self.max_new_tokens),
            )

        return self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
