"""
Adaptive local-video ingestion for VideoGraph.

Processes already-downloaded videos into the artifact layout used by graph
construction: audio, scene clips, adaptive keyframes, and metadata.
"""

from __future__ import annotations

import inspect
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _run_subprocess(cmd: List[str], timeout_s: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _parse_rate(rate: str) -> float:
    raw = str(rate or "").strip()
    if not raw:
        return 0.0
    if "/" in raw:
        num_str, den_str = raw.split("/", 1)
        try:
            num = float(num_str)
            den = float(den_str)
            return num / den if den else 0.0
        except ValueError:
            return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _filter_kwargs_for_callable(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs for compatibility across PySceneDetect versions."""
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs

    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _ffprobe_video(video_path: Path, timeout_s: int) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "format=duration,size:stream=index,codec_type,width,height,avg_frame_rate,r_frame_rate",
        str(video_path),
    ]
    result = _run_subprocess(cmd, timeout_s=timeout_s)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _detect_raw_scenes(
    video_path: Path,
    fps: float,
    duration_s: float,
    scene_cfg: dict,
) -> List[Tuple[float, float]]:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import AdaptiveDetector
    except Exception as exc:
        raise RuntimeError(
            f"PySceneDetect is required (CPU). Install scenedetect/opencv-headless. Error: {exc}"
        ) from exc

    min_scene_len_s = float(scene_cfg.get("min_scene_len_s", 1.5))
    adaptive_threshold = float(scene_cfg.get("adaptive_threshold", 0.5))
    window_width = int(scene_cfg.get("window_width", 2))
    show_progress = bool(scene_cfg.get("show_progress", False))
    min_scene_duration_s = float(scene_cfg.get("min_scene_duration_s", 0.05))

    min_scene_len_frames = max(1, int(round(min_scene_len_s * max(fps, 1.0))))
    detector_kwargs = {
        "adaptive_threshold": adaptive_threshold,
        "min_scene_len": min_scene_len_frames,
        "window_width": window_width,
    }
    detector_kwargs = _filter_kwargs_for_callable(AdaptiveDetector, detector_kwargs)
    detector = AdaptiveDetector(**detector_kwargs)

    manager = SceneManager()
    manager.add_detector(detector)
    stream = open_video(str(video_path))
    manager.detect_scenes(stream, show_progress=show_progress)
    scene_list = manager.get_scene_list()

    scenes: List[Tuple[float, float]] = []
    for start_tc, end_tc in scene_list:
        start_s = float(start_tc.get_seconds())
        end_s = float(end_tc.get_seconds())
        if end_s - start_s >= min_scene_duration_s:
            scenes.append((start_s, end_s))

    if not scenes and duration_s > 0:
        scenes = [(0.0, duration_s)]
    return scenes


def _extract_audio(
    video_path: Path,
    output_dir: Path,
    has_audio_stream: bool,
    timeout_s: int,
) -> Optional[str]:
    audio_path = output_dir / "audio.wav"
    if not has_audio_stream:
        logger.info("  No audio stream found, skipping audio extraction")
        return None

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    run = _run_subprocess(cmd, timeout_s=timeout_s)
    if run.returncode != 0:
        logger.warning(f"  Audio extraction failed (non-fatal): {run.stderr[:200]}")
        return None
    logger.info(f"  Audio extracted: {audio_path}")
    return str(audio_path.name)


def _extract_scene_clips(
    video_path: Path,
    scenes: List[Tuple[float, float]],
    clips_dir: Path,
    clip_cfg: dict,
    timeout_s: int,
) -> List[dict]:
    clips_dir.mkdir(parents=True, exist_ok=True)
    min_scene_duration_s = float(clip_cfg.get("min_scene_duration_s", 0.05))
    video_codec = clip_cfg.get("video_codec", "libx264")
    audio_codec = clip_cfg.get("audio_codec", "aac")
    codec_preset = clip_cfg.get("codec_preset", "ultrafast")

    clips: List[dict] = []
    for i, (start, end) in enumerate(scenes):
        duration = float(end - start)
        if duration < min_scene_duration_s:
            continue

        clip_id = f"clip_{i:04d}"
        clip_path = clips_dir / f"{clip_id}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            video_codec,
            "-c:a",
            audio_codec,
            "-preset",
            codec_preset,
            str(clip_path),
        ]
        run = _run_subprocess(cmd, timeout_s=timeout_s)
        if run.returncode != 0 or not clip_path.exists():
            logger.warning(f"  Failed to extract {clip_id}: {run.stderr[:200]}")
            continue

        clips.append(
            {
                "clip_id": clip_id,
                "path": str(clip_path.relative_to(clips_dir.parent)),
                "start": start,
                "end": end,
                "duration": duration,
                "keyframes": [],
            }
        )
    return clips


def _num_keyframes_for_clip(duration_s: float, keyframe_cfg: dict) -> int:
    short_clip_max_s = float(keyframe_cfg.get("short_clip_max_s", 8.0))
    medium_clip_max_s = float(keyframe_cfg.get("medium_clip_max_s", 20.0))
    short_clip_frames = int(keyframe_cfg.get("short_clip_frames", 1))
    medium_clip_frames = int(keyframe_cfg.get("medium_clip_frames", 2))
    long_clip_frames = int(keyframe_cfg.get("long_clip_frames", 3))

    if duration_s <= short_clip_max_s:
        return max(1, short_clip_frames)
    if duration_s <= medium_clip_max_s:
        return max(1, medium_clip_frames)
    return max(1, long_clip_frames)


def _extract_keyframes_for_clip(
    clip_file: Path,
    clip: dict,
    keyframes_dir: Path,
    keyframe_cfg: dict,
    timeout_s: int,
) -> List[dict]:
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    clip_duration = float(clip.get("duration", 0.0))
    if clip_duration <= 0:
        return []

    def _extract_frame_at(rel_t: float, frame_path: Path) -> bool:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{rel_t:.3f}",
            "-i",
            str(clip_file),
            "-frames:v",
            "1",
            str(frame_path),
        ]
        run = _run_subprocess(cmd, timeout_s=timeout_s)
        return run.returncode == 0 and frame_path.exists()

    n_frames = _num_keyframes_for_clip(clip_duration, keyframe_cfg)
    positions = [((i + 1) / (n_frames + 1)) * clip_duration for i in range(n_frames)]

    clip_id = clip["clip_id"]
    clip_start = float(clip.get("start", 0.0))
    keyframes: List[dict] = []

    for i, rel_t in enumerate(positions, start=1):
        frame_id = f"{clip_id}_kf_{i:02d}"
        frame_path = keyframes_dir / f"{frame_id}.jpg"
        if not _extract_frame_at(rel_t, frame_path):
            logger.warning(f"  Failed to extract keyframe {frame_id} at t={rel_t:.3f}s")
            continue

        keyframes.append(
            {
                "frame_id": frame_id,
                "path": str(frame_path.relative_to(keyframes_dir.parent)),
                "timestamp": clip_start + rel_t,
                "clip_relative_time": rel_t,
            }
        )

    # Add one boundary keyframe near clip end with retry offsets.
    eps = min(max(0.04, 0.02 * clip_duration), 0.20)
    boundary_candidates: List[float] = []
    seen_candidates = set()
    for mult in (1, 2, 3):
        candidate = max(0.0, clip_duration - (mult * eps))
        rounded = round(candidate, 6)
        if rounded in seen_candidates:
            continue
        seen_candidates.add(rounded)
        boundary_candidates.append(candidate)

    boundary_frame_id = f"{clip_id}_kf_{n_frames + 1:02d}"
    boundary_path = keyframes_dir / f"{boundary_frame_id}.jpg"
    boundary_added = False
    for rel_t in boundary_candidates:
        if _extract_frame_at(rel_t, boundary_path):
            keyframes.append(
                {
                    "frame_id": boundary_frame_id,
                    "path": str(boundary_path.relative_to(keyframes_dir.parent)),
                    "timestamp": clip_start + rel_t,
                    "clip_relative_time": rel_t,
                }
            )
            boundary_added = True
            break
    if not boundary_added:
        logger.warning(
            f"  Failed to extract boundary keyframe {boundary_frame_id} "
            f"for {clip_id} after {len(boundary_candidates)} attempts"
        )

    # Strong fallback: try middle frame if sampling failed.
    if not keyframes:
        fallback_rel_t = clip_duration / 2.0
        frame_id = f"{clip_id}_kf_01"
        frame_path = keyframes_dir / f"{frame_id}.jpg"
        if _extract_frame_at(fallback_rel_t, frame_path):
            keyframes.append(
                {
                    "frame_id": frame_id,
                    "path": str(frame_path.relative_to(keyframes_dir.parent)),
                    "timestamp": clip_start + fallback_rel_t,
                    "clip_relative_time": fallback_rel_t,
                }
            )

    return keyframes


def process_local_video_adaptive(
    video_path: str,
    output_dir: str,
    video_id: Optional[str] = None,
    config: Optional[dict] = None,
) -> dict:
    """
    Process a local video using raw PySceneDetect scene clipping.

    Output structure:
    - clips/ (scene clips)
    - keyframes/ (adaptive representative frames per clip)
    - audio.wav (optional)
    - metadata.json (video metadata, scenes, clips, and keyframes)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if config is None:
        config = {}

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if video_id is None:
        video_id = video_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    keyframes_dir = output_dir / "keyframes"
    clips_dir.mkdir(exist_ok=True)
    keyframes_dir.mkdir(exist_ok=True)

    ingestion_cfg = config.get("video_ingestion", {})
    scene_cfg = ingestion_cfg.get("scene_detection", {})
    clip_cfg = ingestion_cfg.get("clip_extraction", {})
    keyframe_cfg = ingestion_cfg.get("adaptive_keyframes", {})
    timeout_s = int(clip_cfg.get("ffmpeg_timeout_s", 1800))

    probe = _ffprobe_video(video_path, timeout_s=timeout_s)
    fmt = probe.get("format", {})
    duration = float(fmt.get("duration") or 0.0)
    file_size = int(fmt.get("size") or 0)

    width, height, fps = 0, 0, 0.0
    has_audio = False
    for stream in probe.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video" and width == 0:
            width = int(stream.get("width", 0) or 0)
            height = int(stream.get("height", 0) or 0)
            fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
        elif codec_type == "audio":
            has_audio = True

    logger.info(f"Processing local video: {video_id} ({duration:.1f}s, {width}x{height})")

    audio_rel = _extract_audio(
        video_path=video_path,
        output_dir=output_dir,
        has_audio_stream=has_audio,
        timeout_s=timeout_s,
    )

    scenes = _detect_raw_scenes(
        video_path=video_path,
        fps=fps,
        duration_s=duration,
        scene_cfg=scene_cfg,
    )
    logger.info(f"  Detected {len(scenes)} scenes")

    clips = _extract_scene_clips(
        video_path=video_path,
        scenes=scenes,
        clips_dir=clips_dir,
        clip_cfg=clip_cfg,
        timeout_s=timeout_s,
    )
    if not clips:
        raise RuntimeError(f"No clips extracted for video {video_id}")
    logger.info(f"  Extracted {len(clips)} clips")

    total_kf = 0
    for clip in clips:
        clip_path = output_dir / clip["path"]
        keyframes = _extract_keyframes_for_clip(
            clip_file=clip_path,
            clip=clip,
            keyframes_dir=keyframes_dir,
            keyframe_cfg=keyframe_cfg,
            timeout_s=timeout_s,
        )
        clip["keyframes"] = keyframes
        total_kf += len(keyframes)
    logger.info(f"  Extracted {total_kf} adaptive keyframes")

    metadata = {
        "metadata": {
            "video_id": video_id,
            "title": video_path.stem,
            "duration": duration,
            "width": width,
            "height": height,
            "fps": fps,
            "source_path": str(video_path),
            "file_size_bytes": file_size,
            "processed_at": datetime.now().isoformat(),
        },
        "audio_path": audio_rel,
        "scenes": [{"start": s, "end": e} for (s, e) in scenes],
        "clips": clips,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Adaptive video ingestion complete. Output: {output_dir}")
    return metadata



