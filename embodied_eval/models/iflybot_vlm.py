from __future__ import annotations

import os
from typing import Optional, Union

from embodied_eval.common.registry import register_model
from embodied_eval.models.internvl3 import InternVL3


@register_model("iflybot_vlm", "iflybotvlm")
class IFlyBotVLM(InternVL3):
    """
    iFlyBotVLM (fine-tuned from InternVL3)
    "https://huggingface.co/iFlyBot/iFlyBotVLM"
    """

    def __init__(
        self,
        model_name_or_path: str = "/home/tanghyyy/data/iFlyBotVLM",
        use_flash_attn: str = False,
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
        num_frame: Optional[int] = 16,
        max_num: Optional[int] = 12,
        **kwargs,
    ) -> None:
        # Keep behavior identical to InternVL3, only default to the local path.
        # If user passes a HF repo id (e.g. iFlyBot/iFlyBotVLM), it will still work.
        if model_name_or_path == "/home/tanghyyy/data/iFlyBotVLM" and not os.path.isdir(model_name_or_path):
            model_name_or_path = "iFlyBot/iFlyBotVLM"

        super().__init__(
            model_name_or_path=model_name_or_path,
            use_flash_attn=use_flash_attn,
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
            num_frame=num_frame,
            max_num=max_num,
            **kwargs,
        )

