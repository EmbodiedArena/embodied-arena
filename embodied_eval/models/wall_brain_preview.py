from __future__ import annotations

from typing import Optional, Union

from embodied_eval.common.registry import register_model
from embodied_eval.models.qwen3_5 import Qwen3_5


@register_model("wall_brain_preview")
class WallBrainPreview(Qwen3_5):
    """
    WALL BRAIN Preview (initialized from Qwen3.5)
    """

    def __init__(
        self,
        model_name_or_path: str = "./path/to/model",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "cuda",
        max_length: Optional[int] = 2048,
        batch_size: Optional[Union[int, str]] = 1,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        num_beams: int = 1,
        do_sample: bool = True,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        min_pixels: int = 1024,
        max_pixels: int = 16777216,
        total_pixels: Optional[int] = None,
        max_num_frames: int = 768,
        max_frames: Optional[int] = 32,
        fps: Optional[float] = 2,
        interleave_visuals: Optional[bool] = False,
        enable_thinking: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Keep behavior identical to Qwen3.5, only default to the local path.
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            device_map=device_map,
            max_length=max_length,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=do_sample,
            use_cache=use_cache,
            attn_implementation=attn_implementation,
            system_prompt=system_prompt,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
            max_num_frames=max_num_frames,
            max_frames=max_frames,
            fps=fps,
            interleave_visuals=interleave_visuals,
            enable_thinking=enable_thinking,
            reasoning_prompt=reasoning_prompt,
            **kwargs,
        )
