from __future__ import annotations

import os
from typing import Optional, Union

from embodied_eval.common.registry import register_model
from embodied_eval.models.qwen3_vl import Qwen3_VL

@register_model("embodied_vlm", "embodiedvlm")
class EmbodiedVLM(Qwen3_VL):
    """
    Embodied-VLM (initialized from Qwen3-VL)
    "https://huggingface.co/IffYuan/Embodied-VLM-8B-RFT-0307"
    """

    def __init__(
            self,
            model_name_or_path: str = "IffYuan/Embodied-VLM-8B-RFT-0307",
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
        # Keep behavior identical to InternVL3, only default to the local path.
        # If user passes a HF repo id (e.g. iFlyBot/iFlyBotVLM), it will still work.
        if model_name_or_path == "/your/path/to/embodied-eval-main/embodied_eval/data/Embodied-VLM-8B-RFT-0307" and not os.path.isdir(model_name_or_path):
            model_name_or_path = "IffYuan/Embodied-VLM-8B-RFT-0307"

        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            device_map=device_map,
            max_length=max_length,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            num_beams=num_beams,
            use_cache=use_cache,
            system_prompt=system_prompt,
            use_flash_attention_2=use_flash_attention_2,
            max_num_frames=max_num_frames,
            fps=fps,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            interleave_visuals=interleave_visuals,
            **kwargs,
        )
