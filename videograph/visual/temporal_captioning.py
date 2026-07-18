"""
Visual captioning with optional previous-clip state change output.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from videograph.cache.openai_cache import get_cache
from videograph.utils import (
    get_openai_client,
    normalize_text,
    resolve_model_name,
    sanitize_entity_strings,
    sanitize_state_change_text,
)

logger = logging.getLogger(__name__)


@dataclass
class TemporalVisualAnalysis:
    """Analysis of a clip, including optional state change from previous clip."""

    clip_id: str
    start: float
    end: float
    visual_description: str
    state_change_from_previous: str
    detected_entities: List[str]
    scene_type: str
    keyframes_analyzed: List[str]
    has_text: bool = False  # captioner's flag: are there readable words on screen worth OCR?
    ocr_text: Optional[str] = None


class TemporalVisualCaptioner:
    """Generates clip-level visual analysis with previous-clip change awareness."""

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
        "other",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        prompt_style: str = "detailed",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        cache_enabled: bool = True,
    ):
        self.client = get_openai_client(api_key)
        self.model = resolve_model_name(model, "vision")
        self.prompt_style = prompt_style
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = get_cache() if cache_enabled else None

    def _encode_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _hash_file(self, path: Path) -> str:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _get_image_media_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

    def _build_prompt(self, include_previous_reference: bool) -> str:
        scene_types = "|".join(self.SCENE_TYPES)
        previous_context = ""
        if include_previous_reference:
            previous_context = (
                "Input convention:\n"
                "- The FIRST image is a reference frame from the immediately PREVIOUS clip.\n"
                "- The remaining images are keyframes from the CURRENT clip.\n"
                "- Describe only the CURRENT clip in visual_description.\n"
                "- Use the previous reference only to infer state_change_from_previous.\n"
            )

        if self.prompt_style == "concise":
            return (
                "You are generating structured visual evidence for downstream retrieval and video QA.\n"
                "Analyze the provided images and return STRICT JSON with exactly these keys:\n"
                "{\n"
                '  "visual_description": "One short factual sentence about the CURRENT clip with concrete subject-action-object detail",\n'
                '  "state_change_from_previous": "One short sentence with meaningful change vs previous clip, or empty string",\n'
                '  "detected_entities": ["flat list of salient entities, including explicit counts when clear (e.g., 2 men, 1 red bus)"],\n'
                f'  "scene_type": "{scene_types}",\n'
                '  "has_text": true or false (true ONLY if readable on-screen text worth extracting is present)\n'
                "}\n\n"
                f"{previous_context}"
                "Rules:\n"
                "- has_text must be true only when clearly readable words/numbers/logos appear on screen; false otherwise.\n"
                "- visual_description must be about the current clip only.\n"
                "- state_change_from_previous must be empty string if there is no clear meaningful change.\n"
                "- Do not use minor camera shifts or tiny pose changes as state change.\n"
                "- detected_entities must be a flat array of strings.\n"
                "- scene_type must be one value from the allowed list.\n"
                "- If the current clip frames show a short action progression, mention that progression in visual_description.\n"
                "Example output for a single clip with within-clip action:\n"
                "{\n"
                "  \"visual_description\": \"A person sits indoors, then lifts a white mug to drink before lowering it and smiling.\",\n"
                "  \"state_change_from_previous\": \"\",\n"
                "  \"detected_entities\": [\"person\", \"white mug\", \"bookshelf\"],\n"
                "  \"scene_type\": \"indoors\"\n"
                "}\n"
                "- No markdown, no comments, no extra keys."
            )

        return (
            "You are generating structured visual evidence for downstream retrieval and multiple-choice video QA.\n"
            "High-value outputs help answer questions about actions, temporal order, causes, and counting.\n"
            "Analyze the provided images and return STRICT JSON with exactly these keys:\n"
            "{\n"
            '  "visual_description": "3-6 factual sentences about the CURRENT clip with concrete actor/action/object details, scene context, and visible text. Including important actions and within-clip changes if any",\n'
            '  "state_change_from_previous": "Short factual summary of meaningful visual change vs previous clip, or empty string",\n'
            '  "detected_entities": ["flat list of salient entities, objects, people, logos, and readable text fragments; include explicit counted items when clear"],\n'
            f'  "scene_type": "{scene_types}",\n'
            '  "has_text": true or false (true ONLY if readable on-screen text is present worth extracting: signs, captions, labels, UI, scoreboards, documents)\n'
            "}\n\n"
            f"{previous_context}"
            "Rules:\n"
            "- has_text must be true only when clearly readable words/numbers/logos appear on screen; false for plain scenes with no meaningful text.\n"
            "- visual_description must describe only the current clip.\n"
            "- visual_description must prioritize discriminative evidence for retrieval: who/what does what, salient objects, and scene context.\n"
            "- If the current clip frames show temporal progression or an action unfolding within the clip, include that progression in visual_description.\n"
            "- Include explicit quantities (use digits) when countable entities or repeated actions are visible and reliable.\n"
            "- Prefer concrete verbs and nouns; avoid vague filler words.\n"
            "- Mention readable on-screen text/logos when visible.\n"
            "- state_change_from_previous must mention only meaningful change and be empty if no meaningful change.\n"
            "- Treat as meaningful change only if people/objects appear-disappear, action phase changes, object location changes, or count changes.\n"
            "- Do not hallucinate changes; if uncertain, return empty string for state_change_from_previous.\n"
            "- Do not use only camera angle/zoom/framing shifts or tiny pose differences as state change.\n"
            "- detected_entities must be a flat array of strings and should include specific useful items (people roles, key objects, logos, readable fragments), not only generic labels.\n"
            "- scene_type must be one allowed value.\n"
            "- Compare the earliest and latest current-clip frames and mention important within-clip changes in objects, actions, or expressions when visible.\n"
            "- Within-clip changes belong in visual_description, not state_change_from_previous.\n"
            "Example output for a single clip with within-clip progression:\n"
            "{\n"
            "  \"visual_description\": \"A woman stands at a kitchen counter with a cutting board and vegetables. At first, she is holding a knife beside the board. In the middle of the clip, she chops the vegetables into smaller pieces. By the end, the chopped pieces are gathered on the board and she moves the knife aside.\",\n"
            "  \"state_change_from_previous\": \"The woman has moved from standing idle at the counter to actively preparing vegetables\",\n"
            "  \"detected_entities\": [\"woman\", \"kitchen counter\", \"cutting board\", \"knife\", \"vegetables\", \"window\"],\n"
            "  \"scene_type\": \"indoors\"\n"
            "}\n"
            "- No markdown, no comments, no extra keys."
        )

    def analyze_frames(
        self,
        frame_paths: List[Path],
        clip_id: str,
        start: float,
        end: float,
        previous_reference_frame: Optional[Path] = None,
    ) -> TemporalVisualAnalysis:
        """
        Analyze current clip frames with optional previous-clip reference frame.
        """
        valid_paths = [Path(p) for p in frame_paths if Path(p).exists()]
        if not valid_paths:
            raise ValueError(f"No valid frame files found: {frame_paths}")

        prev_ref: Optional[Path] = None
        if previous_reference_frame is not None and Path(previous_reference_frame).exists():
            prev_ref = Path(previous_reference_frame)

        # Hybrid keyframes: allow up to 6 current frames (+1 previous reference = 7 images max).
        current_limit = 6
        current_paths = valid_paths[:current_limit]
        if not current_paths:
            raise ValueError(f"No current clip frames selected for clip {clip_id}")

        content = [{"type": "text", "text": self._build_prompt(prev_ref is not None)}]

        if prev_ref:
            prev_media_type = self._get_image_media_type(prev_ref)
            prev_b64 = self._encode_image(prev_ref)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{prev_media_type};base64,{prev_b64}",
                        "detail": "low",
                    },
                }
            )

        for path in current_paths:
            media_type = self._get_image_media_type(path)
            base64_image = self._encode_image(path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_image}",
                        "detail": "high",
                    },
                }
            )

        params = {"temperature": self.temperature, "max_tokens": self.max_tokens, "seed": 0}
        messages = [{"role": "user", "content": content}]

        # Content-addressed cache key: hash the frame PIXELS, not their paths. Identical
        # frames then reuse the cached caption across output dirs / rebuilds, making graph
        # builds deterministic (gpt-4o vision is nondeterministic even at temperature 0,
        # so re-captioning unchanged clips injects ±several answer flips per rebuild).
        cache_key = {
            "current_frames": [self._hash_file(p) for p in current_paths],
            "previous_reference_frame": self._hash_file(prev_ref) if prev_ref else None,
            "prompt_style": self.prompt_style,
            "state_change_enabled": True,
        }

        if self.cache:
            cached = self.cache.get(
                self.model,
                [{"role": "vision", "content": str(cache_key)}],
                params,
            )
            if cached:
                logger.info(f"Using cached VideoGraph visual analysis for clip {clip_id}")
                return self._parse_response(
                    cached,
                    clip_id=clip_id,
                    start=start,
                    end=end,
                    current_frame_paths=current_paths,
                    has_previous_reference=prev_ref is not None,
                )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **params,
            response_format={"type": "json_object"},
        )
        response_text = response.choices[0].message.content

        if self.cache:
            self.cache.set(
                self.model,
                [{"role": "vision", "content": str(cache_key)}],
                params,
                {"text": response_text},
            )

        return self._parse_response(
            {"text": response_text},
            clip_id=clip_id,
            start=start,
            end=end,
            current_frame_paths=current_paths,
            has_previous_reference=prev_ref is not None,
        )

    def _parse_response(
        self,
        response: dict,
        clip_id: str,
        start: float,
        end: float,
        current_frame_paths: List[Path],
        has_previous_reference: bool,
    ) -> TemporalVisualAnalysis:
        try:
            data = json.loads(response.get("text", "{}"))
            if not isinstance(data, dict):
                data = {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse VideoGraph visual response for {clip_id}")
            data = {}

        visual_description = normalize_text(data.get("visual_description", ""))
        state_change = sanitize_state_change_text(data.get("state_change_from_previous", ""))
        if not has_previous_reference:
            state_change = ""

        detected_entities_raw = data.get("detected_entities", [])
        if isinstance(detected_entities_raw, list):
            detected_entities = sanitize_entity_strings(detected_entities_raw)
        else:
            detected_entities = []

        scene_type = str(data.get("scene_type", "other") or "").strip() or "other"
        if scene_type not in self.SCENE_TYPES:
            scene_type = "other"

        has_text = bool(data.get("has_text", False))

        return TemporalVisualAnalysis(
            clip_id=clip_id,
            start=start,
            end=end,
            visual_description=visual_description,
            state_change_from_previous=state_change,
            detected_entities=detected_entities,
            scene_type=scene_type,
            keyframes_analyzed=[Path(p).name for p in current_frame_paths],
            has_text=has_text,
        )


def compose_visual_description_with_state_change(
    visual_description: str,
    state_change_from_previous: str,
) -> str:
    """Append state-change context to visual description for downstream retrieval/embedding."""
    base = str(visual_description or "").strip()
    change = str(state_change_from_previous or "").strip()
    if not change:
        return base

    if base:
        if not base.endswith((".", "!", "?")):
            base = f"{base}."
        return f"{base} Change: {change}"

    return f"Change: {change}"


