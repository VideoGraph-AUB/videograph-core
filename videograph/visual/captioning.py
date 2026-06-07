"""
Visual captioning module using OpenAI Vision API.

For each clip (or representative keyframes), produces:
- visual_description (concise)
- detected_entities (objects, people, on-screen elements)
- scene_type (talking head / slide / outdoors / etc.)
"""

import base64
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
from openai import OpenAI

from ..cache.openai_cache import get_cache

logger = logging.getLogger(__name__)


@dataclass
class VisualAnalysis:
    """Analysis of a visual clip or frame."""
    clip_id: str
    start: float
    end: float
    visual_description: str
    detected_entities: List[str]
    scene_type: str
    keyframes_analyzed: List[str]
    ocr_text: Optional[str] = None  # Added by OCR module


class VisualCaptioner:
    """Generates captions and analysis for video frames/clips using OpenAI Vision."""
    
    SCENE_TYPES = [
        "talking_head",
        "presentation_slide",
        "screencast",
        "outdoors",
        "indoors",
        "diagram",
        "animation",
        "text_heavy",
        "action_scene",
        "interview",
        "product_demo",
        "other"
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        prompt_style: str = "detailed",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        cache_enabled: bool = True
    ):
        """
        Initialize the visual captioner.
        
        Args:
            api_key: OpenAI API key
            model: Vision-capable model name
            prompt_style: "concise" or "detailed"
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            cache_enabled: Whether to cache API calls
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.prompt_style = prompt_style
        self.temperature = temperature
        self.max_tokens = max_tokens
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
    
    def _build_prompt(self) -> str:
        """Build the vision analysis prompt."""
        if self.prompt_style == "concise":
            return """Analyze these frames as one single clip-level scene summary. Provide a JSON response with EXACTLY this schema:
{
  "visual_description": "One clip-level sentence describing what is happening overall",
  "detected_entities": ["entity 1", "entity 2", "entity 3"],
  "scene_type": "talking_head|presentation_slide|screencast|outdoors|indoors|diagram|animation|text_heavy|action_scene|interview|product_demo|other"
}

STRICT OUTPUT RULES:
- visual_description MUST be a single string (NOT an array, NOT an object).
- detected_entities MUST be a flat JSON array of strings only (NOT nested arrays, NOT objects).
- scene_type MUST be one string from the allowed values (NOT an array, NOT an object).
- DO NOT output per-frame structures such as {"frame": 1, ...}.
- DO NOT include markdown, comments, or extra keys.

Be concise and factual."""
        else:
            return """Analyze these frames in detail as one single clip-level scene. Provide a JSON response with EXACTLY this schema:
{
  "visual_description": "A detailed clip-level description including actions, setting, and context across all provided frames",
  "detected_entities": ["flat list of visible objects, people, logos, UI elements, and readable text fragments"],
  "scene_type": "talking_head|presentation_slide|screencast|outdoors|indoors|diagram|animation|text_heavy|action_scene|interview|product_demo|other"
}

Guidelines:
- For talking_head: describe the speaker's appearance, expressions, background
- For presentation_slide: describe the slide content, diagrams, bullet points
- For screencast: describe the software, actions being demonstrated
- For outdoors/indoors: describe the environment, objects, activities
- List ALL text visible on screen as entities
- Be thorough but factual

STRICT OUTPUT RULES:
- visual_description MUST be a single string (NOT an array, NOT an object).
- detected_entities MUST be a flat JSON array of strings only (NOT nested arrays, NOT objects).
- scene_type MUST be one string from the allowed values (NOT an array, NOT an object).
- Aggregate all frames into one clip-level output; do NOT return per-frame fields.
- Forbidden pattern examples: {"frame": 1, ...}, [{"frame": ...}], {"type": ...} inside scene_type.
- DO NOT include markdown, comments, or extra keys."""
    
    def analyze_frames(
        self,
        frame_paths: List[Path],
        clip_id: str,
        start: float,
        end: float
    ) -> VisualAnalysis:
        """
        Analyze one or more frames from a clip.
        
        Args:
            frame_paths: Paths to frame images
            clip_id: ID of the clip these frames belong to
            start: Clip start time
            end: Clip end time
            
        Returns:
            VisualAnalysis object
        """
        if not frame_paths:
            raise ValueError("No frames provided")
        
        # Filter to existing files
        valid_paths = [p for p in frame_paths if Path(p).exists()]
        if not valid_paths:
            raise ValueError(f"No valid frame files found: {frame_paths}")
        
        logger.info(f"Analyzing {len(valid_paths)} frames for clip {clip_id}...")
        
        # Build message content with images
        content = []
        content.append({
            "type": "text",
            "text": self._build_prompt()
        })
        
        for path in valid_paths[:4]:  # Limit to 4 frames to control costs
            path = Path(path)
            media_type = self._get_image_media_type(path)
            base64_image = self._encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_image}",
                    "detail": "low"  # Use low detail to reduce tokens
                }
            })
        
        messages = [{"role": "user", "content": content}]
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # Check cache (using image hashes)
        cache_key = {
            "clip_id": clip_id,
            "frames": [str(p) for p in valid_paths],
            "prompt_style": self.prompt_style
        }
        
        if self.cache:
            cached = self.cache.get(
                self.model,
                [{"role": "vision", "content": str(cache_key)}],
                params
            )
            if cached:
                logger.info(f"Using cached analysis for clip {clip_id}")
                return self._parse_response(cached, clip_id, start, end, valid_paths)
        
        # Call Vision API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **params,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content
        
        # Cache the response
        if self.cache:
            self.cache.set(
                self.model,
                [{"role": "vision", "content": str(cache_key)}],
                params,
                {"text": response_text}
            )
        
        return self._parse_response({"text": response_text}, clip_id, start, end, valid_paths)
    
    def _parse_response(
        self,
        response: dict,
        clip_id: str,
        start: float,
        end: float,
        frame_paths: List[Path]
    ) -> VisualAnalysis:
        """Parse the vision API response."""
        try:
            data = json.loads(response.get("text", "{}"))
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse vision response for {clip_id}")
            data = {}
        
        return VisualAnalysis(
            clip_id=clip_id,
            start=start,
            end=end,
            visual_description=data.get("visual_description", ""),
            detected_entities=data.get("detected_entities", []),
            scene_type=data.get("scene_type", "other"),
            keyframes_analyzed=[Path(p).name for p in frame_paths]
        )


def analyze_clips(
    video_dir: str,
    keyframes_per_clip: int = 3,
    model: str = "gpt-4o",
    prompt_style: str = "detailed",
    temperature: float = 0.3,
    max_parallel: int = 5,
    progress_callback=None
) -> List[dict]:
    """
    Analyze all clips in a video directory with parallel processing.
    
    Args:
        video_dir: Path to video directory (containing metadata.json)
        keyframes_per_clip: Number of keyframes to analyze per clip
        model: Vision model to use
        prompt_style: "concise" or "detailed"
        temperature: Sampling temperature for vision captioning
        max_parallel: Maximum parallel API calls (default 5, recommended for Vision API)
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of visual analysis dictionaries
    """
    from ..processing.parallel import ParallelProcessor
    
    video_dir = Path(video_dir)
    metadata_path = video_dir / "metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {video_dir}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    captioner = VisualCaptioner(model=model, prompt_style=prompt_style, temperature=temperature)
    
    clips = metadata.get("clips", [])
    frames = {f["frame_id"]: f for f in metadata.get("frames", [])}
    
    # Prepare clip data for parallel processing
    clip_tasks = []
    for clip in clips:
        clip_id = clip["clip_id"]
        start = clip["start"]
        end = clip["end"]
        
        # Get keyframes for this clip
        clip_keyframes = clip.get("keyframes", [])
        
        # Select representative frames
        if clip_keyframes:
            # Sample evenly from clip keyframes
            step = max(1, len(clip_keyframes) // keyframes_per_clip)
            selected_frame_ids = clip_keyframes[::step][:keyframes_per_clip]
        else:
            # Find frames by timestamp
            selected_frame_ids = []
            for frame_id, frame in frames.items():
                if start <= frame["timestamp"] < end:
                    selected_frame_ids.append(frame_id)
            # Sample evenly
            if selected_frame_ids:
                step = max(1, len(selected_frame_ids) // keyframes_per_clip)
                selected_frame_ids = selected_frame_ids[::step][:keyframes_per_clip]
        
        # Get frame paths
        frame_paths = []
        for frame_id in selected_frame_ids:
            if frame_id in frames:
                frame_path = video_dir / frames[frame_id]["path"]
                if frame_path.exists():
                    frame_paths.append(frame_path)
        
        if frame_paths:
            clip_tasks.append({
                "clip_id": clip_id,
                "start": start,
                "end": end,
                "frame_paths": frame_paths
            })
    
    if not clip_tasks:
        logger.warning("No clips with frames found for analysis")
        return []
    
    # Define the processing function for each clip
    def process_clip(task):
        try:
            analysis = captioner.analyze_frames(
                task["frame_paths"],
                task["clip_id"],
                task["start"],
                task["end"]
            )
            return asdict(analysis)
        except Exception as e:
            logger.error(f"Failed to analyze clip {task['clip_id']}: {e}")
            return None
    
    # Process clips in parallel
    processor = ParallelProcessor(
        max_workers=max_parallel,
        rate_limit_rpm=60,
        progress_callback=progress_callback
    )
    
    results = processor.process_parallel(
        clip_tasks,
        process_clip,
        stage_name="visual_captioning",
        item_name="clip"
    )
    
    # Filter out failed analyses
    analyses = [r for r in results if r is not None]
    
    # Save analyses
    visual_path = video_dir / "visual.json"
    with open(visual_path, 'w', encoding='utf-8') as f:
        json.dump({"analyses": analyses}, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Visual analysis saved to {visual_path} ({len(analyses)}/{len(clip_tasks)} clips)")
    return analyses


