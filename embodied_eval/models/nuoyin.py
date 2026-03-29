import base64
import os
import time

from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

import requests
from loguru import logger as eval_logger
from PIL import Image

from embodied_eval.common.registry import register_model
from embodied_eval.models import BaseAPIModel

@register_model("nuoyin")
class Nuoyin(BaseAPIModel):
    def __init__(
            self,
            model_name_or_path: str = "KnowinBrain",
            batch_size: Optional[Union[int, str]] = 1,
            max_new_tokens: int = 500,
            temperature: float = 0.7,
            do_sample: bool = False,
            top_p: Optional[float] = None,
            num_beams: int = 1,
            system_prompt: Optional[str] = None,
            timeout: int = 120,
            max_retries: int = 1,
            max_size_in_mb: int = 20,
            max_video_size_in_mb: int = 100,
            **kwargs,
    ) -> None:
        super().__init__()

        # Get API key and base URL from environment variables
        self.api_key = os.getenv("Nuoyin_API_KEY")
        self.base_url = os.getenv("Nuoyin_API_BASE")

        eval_logger.info(f"Base URL: {self.base_url or 'NOT SET'}")
        if self.api_key:
            eval_logger.info(f"API Key: {'*' * min(len(self.api_key), 8)}... (provided)")
        else:
            eval_logger.info("API Key: Not provided (assuming endpoint does not require auth)")
        
        # Store configuration
        self.model_name_or_path = model_name_or_path
        self.batch_size = int(batch_size)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.num_beams = num_beams
        self.system_prompt = system_prompt

        self.max_size_in_mb = max_size_in_mb
        self.max_video_size_in_mb = max_video_size_in_mb
        self.timeout = timeout
        self.max_retries = max_retries

    def encode_image(self, image: Union[Image.Image, str]):
        """Encode image to base64. Uses JPEG format for better performance."""
        max_size = self.max_size_in_mb * 1024 * 1024  # 20MB in bytes
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        # Use JPEG format for better performance (faster encoding, smaller size)
        # Start with quality 90, reduce if needed
        quality = 90
        output_buffer = BytesIO()
        img.save(output_buffer, format="JPEG", quality=quality, optimize=True)
        byte_data = output_buffer.getvalue()

        # If image is too large, resize it while maintaining aspect ratio
        while len(byte_data) > max_size and img.size[0] > 100 and img.size[1] > 100:
            new_size = (int(img.size[0] * 0.75), int(img.size[1] * 0.75))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            output_buffer = BytesIO()
            img.save(output_buffer, format="JPEG", quality=quality, optimize=True)
            byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str
    
    def encode_video(self, video_path: str) -> str:
        """Encode video file to base64 data URL. Optimized for large files."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check file size before reading
        file_size = os.path.getsize(video_path)
        max_size_bytes = self.max_video_size_in_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            eval_logger.warning(
                f"Video file {video_path} is {file_size / (1024*1024):.1f}MB, "
                f"exceeds limit of {self.max_video_size_in_mb}MB. "
                "Consider using a video URL instead or reducing video size."
            )
            # Still proceed, but warn the user
        
        # Determine MIME type first (before reading file)
        ext = Path(video_path).suffix.lower()
        mime_types = {
            ".mp4": "video/mp4",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
        }
        mime_type = mime_types.get(ext, "video/mp4")
        
        # Read and encode video file
        video_name = os.path.basename(video_path)
        file_size_mb = file_size / (1024 * 1024)
        # eval_logger.info(f"Reading video file: {video_name} ({file_size_mb:.1f}MB)...")
        
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        
        # eval_logger.info(f"Video file read, encoding to base64...")
        
        # Use efficient base64 encoding
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        video_data_url = f"data:{mime_type};base64,{video_base64}"
        
        base64_size_mb = len(video_base64) / (1024 * 1024)
        # eval_logger.info(f"Video encoded successfully: {video_name} ({file_size_mb:.1f}MB -> {base64_size_mb:.1f}MB base64)")
        
        return video_data_url
    
    def build_message_content(
        self, 
        question: str, 
        visuals: List[Union[str, Image.Image]]
    ) -> List[dict]:
        """Build message content with support for images and videos.
        All visuals are placed before the text, matching test_multi_image format.
        """
        # All visuals first, then text (simple format like test_multi_image)
        contents = list(visuals) + [question]

        # Convert to OpenAI-style interleaved content
        interleaved = []
        for item in contents:
            if isinstance(item, str):
                # Check if it's a video URL
                if item.startswith("data:video/"):
                    interleaved.append({
                        "type": "video_url",
                        "video_url": {"url": item}
                    })
                # Check if it's an image URL (data URL)
                elif item.startswith("data:image/"):
                    interleaved.append({
                        "type": "image_url",
                        "image_url": {"url": item}
                    })
                # Check if it's a video HTTP/HTTPS URL
                elif (item.startswith("http://") or item.startswith("https://")) and (
                    any(ext in item.lower() for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"])
                ):
                    interleaved.append({
                        "type": "video_url",
                        "video_url": {"url": item}
                    })
                # Check if it's an image HTTP/HTTPS URL
                # Since _prepare_visuals already filters URLs, any HTTP/HTTPS URL here is likely an image
                elif item.startswith("http://") or item.startswith("https://"):
                    interleaved.append({
                        "type": "image_url",
                        "image_url": {"url": item}
                    })
                else:
                    interleaved.append({"type": "text", "text": item})
            elif isinstance(item, Image.Image):
                # This shouldn't happen as _prepare_visuals should have encoded all images
                # But handle it just in case
                base64_img = self.encode_image(item)
                interleaved.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                })
        return interleaved
    
    def _build_payload(self, context, visuals, gen_kwargs):
        payload = {
            "model": self.model_name_or_path,
            "messages": []
        }
        if self.system_prompt:
            payload["messages"].append({
                "role": "system",
                "content": self.system_prompt
            })

        content = self.build_message_content(
            question=context,
            visuals=visuals
        )

        payload["messages"].append({
            "role": "user",
            "content": content
        })

        max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = gen_kwargs.get("temperature", self.temperature)
        payload["max_tokens"] = max_new_tokens
        payload["temperature"] = temperature


        return payload

    def respond(self, context, visuals, **gen_kwargs) -> str:
        """Generate a single response for the given context and visuals."""
        return self._answer(context, visuals, gen_kwargs or {})
    
    def _answer(self, context, visual_list, gen_kwargs):
        """Call Nuoyin/KnowinBrain endpoint using requests."""
        visuals = self._prepare_visuals(visual_list)
        payload = self._build_payload(context, visuals, gen_kwargs)

        if not self.base_url:
            eval_logger.error("Nuoyin_API_BASE is not set; cannot send request.")
            return ""

        # Construct URL: ensure it ends with /chat/completions
        base_url = self.base_url.rstrip("/")
        if not base_url.endswith("/chat/completions"):
            url = f"{base_url}/chat/completions"
        else:
            url = base_url

        response_text = ""
        error_msg = ""

        for attempt in range(self.max_retries):
            try:
                # Direct synchronous HTTP request
                resp = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )

                if resp.status_code != 200:
                    error_msg = f"Non-200 status code: {resp.status_code}, body: {resp.text[:500]}"
                    raise RuntimeError(error_msg)

                data = resp.json()
                if "choices" in data and data["choices"]:
                    response_text = data["choices"][0]["message"]["content"]
                else:
                    error_msg = f"Unexpected response format: {data}"
                    raise RuntimeError(error_msg)

                break
            except requests.exceptions.Timeout as e:
                error_msg = f"Requests Timeout Error: {e}"
                eval_logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed with timeout: {error_msg}")
            except requests.exceptions.RequestException as e:
                error_msg = f"Requests Error: {e}"
                eval_logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed with request error: {error_msg}")
            except Exception as e:
                error_msg = str(e)
                eval_logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed with general error: {error_msg}")

            if attempt == self.max_retries - 1:
                eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
            else:
                eval_logger.info("Retrying after 0.5s...")
                time.sleep(0.5)

        return response_text


    def _prepare_visuals(self, visual_list):
        """Prepare visuals: encode images and videos, return list of visuals.
        Optimized to skip encoding for URLs.
        """
        if visual_list is None:
            return []

        # Helper to check if string is already a URL
        def is_url(s: str) -> bool:
            return s.startswith("http://") or s.startswith("https://") or s.startswith("data:")

        # Helper to check if string is a video file path
        def is_video_file(s: str) -> bool:
            return any(ext in s.lower() for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"])

        # Helper to encode a single visual
        def encode_single_visual(visual):
            if isinstance(visual, Image.Image):
                # Encode PIL Image
                base64_img = self.encode_image(visual)
                return f"data:image/jpeg;base64,{base64_img}"
            elif isinstance(visual, str):
                # If already a URL, use directly
                if is_url(visual):
                    return visual
                # Check if it's a video file
                elif is_video_file(visual):
                    # Direct sync call for video
                    video_data_url = self.encode_video(visual)
                    return video_data_url
                else:
                    # Local image file path, encode it
                    base64_img = self.encode_image(visual)
                    return f"data:image/jpeg;base64,{base64_img}"
            else:
                return visual

        # Handle list of visuals
        if isinstance(visual_list, (list, tuple)):
            visuals = [encode_single_visual(visual) for visual in visual_list]
            return visuals
        # Handle single visual
        else:
            result = encode_single_visual(visual_list)
            return [result]
