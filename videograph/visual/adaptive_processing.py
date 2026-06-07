"""
Visual processing over adaptive per-clip keyframes.

Reads metadata produced by adaptive local-video ingestion and writes visual.json
in the schema expected by graph construction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from videograph.processing.parallel import ParallelProcessor
from videograph.visual.ocr import VisionOCR

from videograph.visual.temporal_captioning import (
    TemporalVisualCaptioner,
    compose_visual_description_with_state_change,
)

logger = logging.getLogger(__name__)


def _load_metadata(video_dir: Path) -> dict:
    metadata_path = video_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {video_dir}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clip_keyframe_paths(video_dir: Path, clip: dict) -> List[Path]:
    keyframes = clip.get("keyframes", []) or []
    paths: List[Path] = []
    for entry in keyframes:
        if isinstance(entry, dict):
            raw_path = entry.get("path")
        else:
            raw_path = str(entry)
        if not raw_path:
            continue
        p = Path(raw_path)
        full = p if p.is_absolute() else (video_dir / p)
        if full.exists():
            paths.append(full)
    return paths


def _save_state_changes_index(video_dir: Path, analyses: List[dict]) -> None:
    """
    Persist clip -> state_change mapping for QA phase.

    This survives cleanup mode where visual.json may be deleted.
    """
    mapping = {}
    for row in analyses:
        clip_id = str(row.get("clip_id", "") or "").strip()
        state_change = str(row.get("state_change_from_previous", "") or "").strip()
        if clip_id and state_change:
            mapping[clip_id] = state_change

    payload = {
        "state_change_by_clip": mapping,
        "non_empty_count": len(mapping),
    }
    state_path = Path(video_dir) / "state_changes.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def analyze_adaptive_clips(
    video_dir: str,
    model: str = "gpt-4o",
    prompt_style: str = "detailed",
    temperature: float = 0.0,
    max_parallel: int = 5,
    append_state_change_to_description: bool = False,
    progress_callback=None,
) -> List[dict]:
    """
    Analyze clips from adaptive keyframes and save visual.json.

    By default, state_change_from_previous is kept as a separate field and is
    not appended into visual_description (to avoid embedding noise).
    """
    video_dir = Path(video_dir)
    metadata = _load_metadata(video_dir)
    clips = metadata.get("clips", [])
    if not clips:
        logger.warning("No clips found in metadata for visual analysis")
        return []

    captioner = TemporalVisualCaptioner(model=model, prompt_style=prompt_style, temperature=temperature,)

    clip_tasks: List[dict] = []
    for clip in clips:
        frame_paths = _clip_keyframe_paths(video_dir, clip)
        if not frame_paths:
            continue
        clip_tasks.append(
            {
                "clip_id": clip.get("clip_id"),
                "start": float(clip.get("start", 0.0)),
                "end": float(clip.get("end", 0.0)),
                "frame_paths": frame_paths,
            }
        )

    # Maintain timeline ordering so previous-reference context is stable.
    clip_tasks.sort(key=lambda task: (task.get("start", 0.0), str(task.get("clip_id", ""))))
    previous_refs_by_clip: Dict[str, Optional[Path]] = {}
    previous_last_keyframe: Optional[Path] = None
    for task in clip_tasks:
        clip_id = str(task.get("clip_id", ""))
        previous_refs_by_clip[clip_id] = previous_last_keyframe
        paths: List[Path] = task.get("frame_paths", [])
        previous_last_keyframe = paths[-1] if paths else previous_last_keyframe

    if not clip_tasks:
        logger.warning("No clips with keyframes found for visual analysis")
        visual_path = video_dir / "visual.json"
        with open(visual_path, "w", encoding="utf-8") as f:
            json.dump({"analyses": []}, f, indent=2, ensure_ascii=False)
        return []

    def process_clip(task):
        try:
            analysis = captioner.analyze_frames(
                frame_paths=task["frame_paths"],
                clip_id=task["clip_id"],
                start=task["start"],
                end=task["end"],
                previous_reference_frame=previous_refs_by_clip.get(str(task["clip_id"])),
            )
            analysis_dict = asdict(analysis)
            if append_state_change_to_description:
                analysis_dict["visual_description"] = compose_visual_description_with_state_change(
                    analysis_dict.get("visual_description", ""),
                    analysis_dict.get("state_change_from_previous", ""),
                )
            return analysis_dict
        except Exception as exc:
            logger.error(f"Failed visual analysis for clip {task['clip_id']}: {exc}")
            return None

    processor = ParallelProcessor(
        max_workers=max_parallel,
        rate_limit_rpm=60,
        progress_callback=progress_callback,
    )
    results = processor.process_parallel(
        clip_tasks,
        process_clip,
        stage_name="visual_captioning",
        item_name="clip",
    )
    analyses = [r for r in results if r is not None]

    visual_path = video_dir / "visual.json"
    with open(visual_path, "w", encoding="utf-8") as f:
        json.dump({"analyses": analyses}, f, indent=2, ensure_ascii=False)
    _save_state_changes_index(video_dir, analyses)

    logger.info(f"Visual analysis saved to {visual_path} ({len(analyses)}/{len(clip_tasks)} clips)")
    return analyses


def _dedupe_lines(texts: List[str]) -> str:
    seen = set()
    ordered: List[str] = []
    for text in texts:
        for line in str(text or "").splitlines():
            line = line.strip()
            if not line:
                continue
            if line in seen:
                continue
            seen.add(line)
            ordered.append(line)
    return "\n".join(ordered)


def update_adaptive_visual_json_with_ocr(
    video_dir: str,
    model: str = "gpt-4o",
    max_parallel: int = 5,
    progress_callback=None,
):
    """
    Run OCR on all selected keyframes for each clip and update visual.json.
    """
    video_dir = Path(video_dir)
    metadata = _load_metadata(video_dir)
    clips = metadata.get("clips", [])
    if not clips:
        logger.warning("No clips found in metadata for OCR")
        return

    ocr = VisionOCR(model=model)

    ocr_tasks = []
    for clip in clips:
        ocr_tasks.append(
            {
                "clip_id": clip.get("clip_id"),
                "start": float(clip.get("start", 0.0)),
                "end": float(clip.get("end", 0.0)),
                "frame_paths": _clip_keyframe_paths(video_dir, clip),
            }
        )

    def process_ocr_task(task):
        texts: List[str] = []
        for frame_path in task["frame_paths"]:
            text = ocr.extract_text(frame_path)
            if text:
                texts.append(text)
        return {
            "clip_id": task["clip_id"],
            "start": task["start"],
            "end": task["end"],
            "ocr_text": _dedupe_lines(texts),
        }

    processor = ParallelProcessor(
        max_workers=max_parallel,
        rate_limit_rpm=60,
        progress_callback=progress_callback,
    )
    results = processor.process_parallel(
        ocr_tasks,
        process_ocr_task,
        stage_name="ocr_extraction",
        item_name="clip",
    )
    ocr_results: Dict[str, dict] = {
        r["clip_id"]: r for r in results if r and r.get("clip_id")
    }

    visual_path = video_dir / "visual.json"
    if visual_path.exists():
        with open(visual_path, "r", encoding="utf-8") as f:
            visual_data = json.load(f)
    else:
        visual_data = {"analyses": []}

    analyses = visual_data.get("analyses", [])
    indexed = {a.get("clip_id"): a for a in analyses if a.get("clip_id")}

    for row in analyses:
        row.setdefault("state_change_from_previous", "")

    # Ensure every clip has an analysis row.
    for clip in clips:
        clip_id = clip.get("clip_id")
        if clip_id in indexed:
            continue
        new_row = {
            "clip_id": clip_id,
            "start": float(clip.get("start", 0.0)),
            "end": float(clip.get("end", 0.0)),
            "visual_description": "",
            "detected_entities": [],
            "scene_type": "unknown",
            "keyframes_analyzed": [],
            "ocr_text": "",
            "state_change_from_previous": "",
        }
        analyses.append(new_row)
        indexed[clip_id] = new_row

    for clip_id, ocr_row in ocr_results.items():
        if clip_id in indexed:
            indexed[clip_id]["ocr_text"] = ocr_row.get("ocr_text", "")

    with open(visual_path, "w", encoding="utf-8") as f:
        json.dump({"analyses": analyses}, f, indent=2, ensure_ascii=False)
    _save_state_changes_index(video_dir, analyses)

    logger.info(f"Updated {visual_path} with OCR text")



