"""
OCR module using OpenAI Vision API.

Extracts on-screen text from video frames using GPT-4o Vision
instead of local Tesseract to minimize dependencies.
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from ..cache.openai_cache import get_cache
from ..utils import get_openai_client, resolve_model_name, sanitize_ocr_text

logger = logging.getLogger(__name__)


class VisionOCR:
    """Extracts text from images using OpenAI Vision API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        cache_enabled: bool = True
    ):
        self.client = get_openai_client(api_key)
        self.model = resolve_model_name(model, "vision")
        self.cache = get_cache() if cache_enabled else None
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode an image file as base64."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    
    def _get_image_media_type(self, path: Path) -> str:
        """Get the MIME type for an image."""
        suffix = path.suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }.get(suffix, "image/jpeg")
    
    def extract_text(self, image_path: Path) -> str:
        """
        Extract text from an image using Vision API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text
        """
        image_path = Path(image_path)
        if not image_path.exists():
            return ""
        
        # Build message
        media_type = self._get_image_media_type(image_path)
        base64_image = self._encode_image(image_path)
        
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Extract ALL readable text visible in this image. "
                        "Return ONLY the extracted text as plain text, with no explanation, no apology, "
                        "no commentary, and no surrounding quotes. "
                        "If there is no readable text, return exactly an empty response. "
                        "Do NOT say things like 'I can't extract text', 'no text found', or similar."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_image}",
                        "detail": "low"
                    }
                }
            ]
        }]
        
        params = {"temperature": 0, "max_tokens": 500}
        
        # Check cache
        cache_key = {"path": str(image_path), "action": "ocr"}
        if self.cache:
            cached = self.cache.get(
                self.model,
                [{"role": "ocr", "content": str(image_path)}],
                cache_key
            )
            if cached:
                return cached.get("text", "")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **params
            )
            text = sanitize_ocr_text(response.choices[0].message.content)
            
            # Cache result
            if self.cache:
                self.cache.set(
                    self.model,
                    [{"role": "ocr", "content": str(image_path)}],
                    cache_key,
                    {"text": text}
                )
            
            return text
        except Exception as e:
            logger.warning(f"OCR failed for {image_path}: {e}")
            return ""


def extract_ocr_for_clips(
    video_dir: str,
    keyframes_per_clip: int = 1,
    model: str = "gpt-4o",
    max_parallel: int = 5,
    progress_callback=None
) -> Dict[str, str]:
    """
    Extract OCR text for all clips using OpenAI Vision with parallel processing.
    
    Args:
        video_dir: Path to video directory
        keyframes_per_clip: Number of keyframes to OCR per clip
        model: OpenAI model to use
        max_parallel: Maximum parallel API calls
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary mapping clip_id to extracted OCR text
    """
    from ..processing.parallel import ParallelProcessor
    
    video_dir = Path(video_dir)
    metadata_path = video_dir / "metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {video_dir}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    frames = {f["frame_id"]: f for f in metadata.get("frames", [])}
    clips = metadata.get("clips", [])
    
    ocr = VisionOCR(model=model)
    
    # Prepare OCR tasks (one per clip, with selected keyframes)
    ocr_tasks = []
    for clip in clips:
        clip_id = clip["clip_id"]
        clip_keyframes = clip.get("keyframes", [])
        
        # Select keyframes for OCR (fewer to minimize API calls)
        if clip_keyframes:
            step = max(1, len(clip_keyframes) // keyframes_per_clip)
            selected = clip_keyframes[::step][:keyframes_per_clip]
        else:
            selected = []
        
        # Get frame paths
        frame_paths = []
        for frame_id in selected:
            if frame_id in frames:
                frame_path = video_dir / frames[frame_id]["path"]
                if frame_path.exists():
                    frame_paths.append(frame_path)
        
        ocr_tasks.append({
            "clip_id": clip_id,
            "frame_paths": frame_paths
        })
    
    # Define processing function
    def process_ocr_task(task):
        texts = []
        for frame_path in task["frame_paths"]:
            text = ocr.extract_text(frame_path)
            if text:
                texts.append(text)
        
        # Combine and deduplicate
        if texts:
            all_lines = []
            for text in texts:
                for line in text.split('\n'):
                    line = line.strip()
                    if line and line not in all_lines:
                        all_lines.append(line)
            return {"clip_id": task["clip_id"], "ocr_text": '\n'.join(all_lines)}
        return {"clip_id": task["clip_id"], "ocr_text": ""}
    
    # Process in parallel
    processor = ParallelProcessor(
        max_workers=max_parallel,
        rate_limit_rpm=60,
        progress_callback=progress_callback
    )
    
    results = processor.process_parallel(
        ocr_tasks,
        process_ocr_task,
        stage_name="ocr_extraction",
        item_name="clip"
    )
    
    # Build results dict
    ocr_results = {}
    for result in results:
        if result:
            ocr_results[result["clip_id"]] = result["ocr_text"]
    
    logger.info(f"Extracted OCR for {len(ocr_results)} clips")
    return ocr_results


def update_visual_json_with_ocr(
    video_dir: str,
    model: str = "gpt-4o",
    max_parallel: int = 5,
    progress_callback=None
):
    """
    Run OCR and update visual.json with extracted text.
    
    Args:
        video_dir: Path to video directory
        model: OpenAI model to use
        max_parallel: Maximum parallel API calls
        progress_callback: Optional callback for progress updates
    """
    video_dir = Path(video_dir)
    visual_path = video_dir / "visual.json"
    
    # Extract OCR text with parallel processing
    ocr_results = extract_ocr_for_clips(
        video_dir,
        model=model,
        max_parallel=max_parallel,
        progress_callback=progress_callback
    )
    
    # Update visual.json
    if visual_path.exists():
        with open(visual_path, 'r', encoding='utf-8') as f:
            visual_data = json.load(f)
        
        for analysis in visual_data.get("analyses", []):
            clip_id = analysis.get("clip_id")
            if clip_id in ocr_results:
                analysis["ocr_text"] = ocr_results[clip_id]
        
        with open(visual_path, 'w', encoding='utf-8') as f:
            json.dump(visual_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Updated {visual_path} with OCR text")
    else:
        # Create visual.json with just OCR data
        metadata_path = video_dir / "metadata.json"
        analyses = []
        
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            for clip in metadata.get("clips", []):
                clip_id = clip["clip_id"]
                analyses.append({
                    "clip_id": clip_id,
                    "start": clip["start"],
                    "end": clip["end"],
                    "visual_description": "",
                    "detected_entities": [],
                    "scene_type": "unknown",
                    "keyframes_analyzed": [],
                    "ocr_text": ocr_results.get(clip_id, "")
                })
        
        with open(visual_path, 'w', encoding='utf-8') as f:
            json.dump({"analyses": analyses}, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created {visual_path} with OCR text")

