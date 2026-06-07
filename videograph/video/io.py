"""
YouTube video ingestion module.

Handles:
- Downloading YouTube videos via yt-dlp
- Extracting audio to WAV
- Extracting frames (configurable FPS + simple keyframe detection)
- Segmenting clips (fixed-length, no heavy CV dependencies)

Note: Uses minimal dependencies - no OpenCV or PySceneDetect.
All heavy processing is done via ffmpeg which is required for video handling.
"""

import json
import logging
import os
import subprocess
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata for a downloaded video."""
    video_id: str
    title: str
    description: str
    duration: float  # seconds
    upload_date: str
    channel: str
    url: str
    download_time: str
    width: int
    height: int
    fps: float


@dataclass
class FrameInfo:
    """Information about an extracted frame."""
    frame_id: str
    path: str
    timestamp: float  # seconds
    is_keyframe: bool
    scene_id: Optional[int] = None


@dataclass
class ClipInfo:
    """Information about an extracted clip."""
    clip_id: str
    path: str
    start: float  # seconds
    end: float  # seconds
    duration: float
    scene_id: Optional[int] = None
    keyframes: List[str] = None  # list of frame_ids


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def detect_scene_changes_ffmpeg(video_path: Path, threshold: float = 0.3) -> List[float]:
    """
    Detect scene changes using ffmpeg's scene detection filter.
    
    This is a lightweight alternative to PySceneDetect that uses
    ffmpeg's built-in scene detection capabilities.
    
    Args:
        video_path: Path to the video file
        threshold: Scene change threshold (0.0-1.0, lower = more sensitive)
        
    Returns:
        List of timestamps where scene changes occur
    """
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Parse timestamps from showinfo output
        timestamps = []
        for line in result.stderr.split('\n'):
            if 'pts_time' in line:
                match = re.search(r'pts_time:([0-9.]+)', line)
                if match:
                    timestamps.append(float(match.group(1)))
        
        return timestamps
    except Exception as e:
        logger.warning(f"Scene detection failed: {e}")
        return []


class VideoDownloader:
    """Downloads and processes YouTube videos."""
    
    def __init__(
        self,
        output_base: str = "data/videos",
        frame_fps: float = 1.0,
        use_scene_detect: bool = True,
        scene_threshold: float = 0.3,
        clip_length: float = 15.0,
        use_scene_clips: bool = True,
        min_clip_length: float = 5.0,
        max_clip_length: float = 30.0
    ):
        """
        Initialize the video downloader.
        
        Args:
            output_base: Base directory for video outputs
            frame_fps: Frames per second for sampling
            use_scene_detect: Whether to use scene change detection (via ffmpeg)
            scene_threshold: Threshold for scene detection (0.0-1.0)
            clip_length: Fixed clip length in seconds (fallback)
            use_scene_clips: Whether to use scene-based clip segmentation
            min_clip_length: Minimum clip length in seconds
            max_clip_length: Maximum clip length in seconds
        """
        self.output_base = Path(output_base)
        self.frame_fps = frame_fps
        self.use_scene_detect = use_scene_detect
        self.scene_threshold = scene_threshold
        self.clip_length = clip_length
        self.use_scene_clips = use_scene_clips
        self.min_clip_length = min_clip_length
        self.max_clip_length = max_clip_length
    
    def _get_output_dir(self, video_id: str) -> Path:
        """Get the output directory for a video."""
        output_dir = self.output_base / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "frames").mkdir(exist_ok=True)
        (output_dir / "clips").mkdir(exist_ok=True)
        return output_dir
    
    def download_video(self, url: str) -> Tuple[Path, VideoMetadata]:
        """
        Download a YouTube video.
        
        Args:
            url: YouTube URL
            
        Returns:
            Tuple of (video_path, metadata)
        """
        video_id = extract_video_id(url)
        output_dir = self._get_output_dir(video_id)
        video_path = output_dir / "video.mp4"
        
        logger.info(f"Downloading video {video_id}...")
        logger.info(f"Output directory: {output_dir}")
        
        # Download video with yt-dlp (with progress)
        cmd = [
            "yt-dlp",
            "-f", (
                "bestvideo[vcodec^=avc1][ext=mp4][height<=720]+bestaudio[ext=m4a]/"
                "best[ext=mp4][vcodec^=avc1][height<=720]/"
                "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/"
                "best[ext=mp4][height<=720]/best"
            ),
            "-o", str(video_path),
            "--write-info-json",
            "--no-playlist",
            "--merge-output-format", "mp4",
            "--progress",
            "--newline",
            url
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        # Stream output for progress visibility
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            line = line.strip()
            if line:
                # Log download progress lines
                if '[download]' in line or 'Downloading' in line or '%' in line:
                    logger.info(f"yt-dlp: {line}")
                elif 'Merging' in line or 'Deleting' in line:
                    logger.info(f"yt-dlp: {line}")
        
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"yt-dlp failed with code {process.returncode}")
        
        # Load metadata from info JSON
        info_path = output_dir / "video.info.json"
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
        else:
            # Fallback: get metadata via yt-dlp
            cmd = ["yt-dlp", "-j", url]
            result = subprocess.run(cmd, capture_output=True, text=True)
            info = json.loads(result.stdout)
        
        metadata = VideoMetadata(
            video_id=video_id,
            title=info.get("title", ""),
            description=info.get("description", ""),
            duration=info.get("duration", 0),
            upload_date=info.get("upload_date", ""),
            channel=info.get("channel", info.get("uploader", "")),
            url=url,
            download_time=datetime.now().isoformat(),
            width=info.get("width", 0),
            height=info.get("height", 0),
            fps=info.get("fps", 30)
        )
        
        logger.info(f"Downloaded: {metadata.title} ({metadata.duration:.1f}s)")
        return video_path, metadata
    
    def extract_audio(self, video_path: Path) -> Path:
        """
        Extract audio from video as WAV.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the extracted audio file
        """
        audio_path = video_path.parent / "audio.wav"
        
        logger.info("Extracting audio from video...")
        logger.info(f"  Input: {video_path}")
        logger.info(f"  Output: {audio_path}")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",  # no video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", "16000",  # 16kHz sample rate (Whisper optimal)
            "-ac", "1",  # mono
            "-progress", "pipe:1",
            str(audio_path)
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed: {stderr}")
        
        # Get file size
        size_mb = audio_path.stat().st_size / (1024 * 1024)
        logger.info(f"Audio extracted: {audio_path} ({size_mb:.1f} MB)")
        return audio_path
    
    def detect_scenes(self, video_path: Path) -> List[Tuple[float, float]]:
        """
        Detect scene boundaries using ffmpeg's scene filter.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of (start, end) tuples for each scene
        """
        if not self.use_scene_detect:
            return []
        
        logger.info("Detecting scenes via ffmpeg...")
        
        scene_timestamps = detect_scene_changes_ffmpeg(video_path, self.scene_threshold)
        
        if not scene_timestamps:
            logger.info("No scene changes detected")
            return []
        
        # Get video duration
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = float(result.stdout.strip()) if result.stdout.strip() else 0
        
        # Convert timestamps to scene boundaries
        scenes = []
        prev_time = 0.0
        
        for timestamp in scene_timestamps:
            if timestamp > prev_time:
                scenes.append((prev_time, timestamp))
            prev_time = timestamp
        
        # Add final scene
        if prev_time < duration:
            scenes.append((prev_time, duration))
        
        logger.info(f"Detected {len(scenes)} scenes")
        return scenes
    
    def extract_frames(
        self, 
        video_path: Path, 
        scenes: List[Tuple[float, float]],
        duration: float
    ) -> List[FrameInfo]:
        """
        Extract frames from video at regular intervals.
        
        Args:
            video_path: Path to the video file
            scenes: List of scene boundaries (for keyframe marking)
            duration: Video duration in seconds
            
        Returns:
            List of FrameInfo objects
        """
        output_dir = video_path.parent / "frames"
        frames = []
        
        expected_frames = int(duration * self.frame_fps)
        logger.info(f"Extracting frames at {self.frame_fps} fps (expecting ~{expected_frames} frames)...")
        logger.info(f"  Output: {output_dir}")
        
        # Extract frames at regular intervals using ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={self.frame_fps}",
            "-frame_pts", "1",
            str(output_dir / "frame_%06d.jpg")
        ]
        
        logger.info("Running ffmpeg for frame extraction...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr}")
        
        logger.info("Frame extraction complete, processing frame metadata...")
        
        # Create FrameInfo for each extracted frame
        frame_interval = 1.0 / self.frame_fps
        frame_files = sorted(output_dir.glob("frame_*.jpg"))
        
        # Get scene start times for keyframe detection
        scene_starts = {s[0] for s in scenes} if scenes else set()
        
        for i, frame_file in enumerate(frame_files):
            timestamp = i * frame_interval
            if timestamp > duration:
                break
            
            frame_id = f"frame_{i:06d}"
            
            # Mark as keyframe if near a scene start
            is_keyframe = any(
                abs(timestamp - start) < frame_interval
                for start in scene_starts
            )
            
            # Find scene ID
            scene_id = None
            for j, (start, end) in enumerate(scenes):
                if start <= timestamp < end:
                    scene_id = j
                    break
            
            frames.append(FrameInfo(
                frame_id=frame_id,
                path=str(frame_file.relative_to(video_path.parent)),
                timestamp=timestamp,
                is_keyframe=is_keyframe,
                scene_id=scene_id
            ))
        
        keyframe_count = sum(1 for f in frames if f.is_keyframe)
        logger.info(f"Extracted {len(frames)} frames ({keyframe_count} keyframes)")
        return frames
    
    def extract_clips(
        self,
        video_path: Path,
        scenes: List[Tuple[float, float]],
        duration: float
    ) -> List[ClipInfo]:
        """
        Extract video clips based on scenes or fixed intervals.
        
        Args:
            video_path: Path to the video file
            scenes: List of scene boundaries
            duration: Video duration in seconds
            
        Returns:
            List of ClipInfo objects
        """
        output_dir = video_path.parent / "clips"
        clips = []
        
        # Determine clip boundaries
        if self.use_scene_clips and scenes:
            # Merge short scenes, split long ones
            clip_boundaries = []
            
            for start, end in scenes:
                scene_duration = end - start
                
                if scene_duration < self.min_clip_length:
                    # Too short, will be merged or skipped
                    if clip_boundaries and clip_boundaries[-1][1] == start:
                        # Extend previous clip
                        prev_start, _ = clip_boundaries.pop()
                        clip_boundaries.append((prev_start, end))
                    continue
                elif scene_duration > self.max_clip_length:
                    # Too long, split into smaller clips
                    for t in range(int(start), int(end), int(self.clip_length)):
                        clip_start = float(t)
                        clip_end = min(clip_start + self.clip_length, end)
                        if clip_end - clip_start >= self.min_clip_length:
                            clip_boundaries.append((clip_start, clip_end))
                else:
                    clip_boundaries.append((start, end))
        else:
            # Fixed-length clips
            clip_boundaries = []
            for t in range(0, int(duration), int(self.clip_length)):
                start = float(t)
                end = min(start + self.clip_length, duration)
                if end - start >= self.min_clip_length:
                    clip_boundaries.append((start, end))
        
        logger.info(f"Extracting {len(clip_boundaries)} clips...")
        logger.info(f"  Output: {output_dir}")
        
        # Extract each clip
        for i, (start, end) in enumerate(clip_boundaries):
            clip_id = f"clip_{i:04d}"
            clip_path = output_dir / f"{clip_id}.mp4"
            
            if (i + 1) % 5 == 0 or i == 0:
                logger.info(f"  Extracting clip {i+1}/{len(clip_boundaries)} ({start:.1f}s - {end:.1f}s)")
            
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", str(video_path),
                "-t", str(end - start),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "ultrafast",
                str(clip_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Failed to extract clip {clip_id}: {result.stderr}")
                continue
            
            clips.append(ClipInfo(
                clip_id=clip_id,
                path=str(clip_path.relative_to(video_path.parent)),
                start=start,
                end=end,
                duration=end - start,
                scene_id=i if scenes else None,
                keyframes=[]
            ))
        
        logger.info(f"Extracted {len(clips)} clips")
        return clips
    
    def process_video(self, url: str) -> dict:
        """
        Full pipeline: download, extract audio, frames, and clips.
        
        Args:
            url: YouTube URL
            
        Returns:
            Dictionary with all metadata and file paths
        """
        logger.info("=" * 60)
        logger.info("STARTING VIDEO PROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Download video
        logger.info("[1/5] Downloading video...")
        video_path, metadata = self.download_video(url)
        output_dir = video_path.parent
        logger.info(f"[1/5] Download complete: {metadata.title}")
        
        # Step 2: Extract audio
        logger.info("[2/5] Extracting audio...")
        audio_path = self.extract_audio(video_path)
        logger.info("[2/5] Audio extraction complete")
        
        # Step 3: Detect scenes (lightweight via ffmpeg)
        logger.info("[3/5] Detecting scenes...")
        scenes = self.detect_scenes(video_path)
        logger.info(f"[3/5] Scene detection complete: {len(scenes)} scenes")
        
        # Step 4: Extract frames
        logger.info("[4/5] Extracting frames...")
        frames = self.extract_frames(video_path, scenes, metadata.duration)
        logger.info(f"[4/5] Frame extraction complete: {len(frames)} frames")
        
        # Step 5: Extract clips
        logger.info("[5/5] Extracting clips...")
        clips = self.extract_clips(video_path, scenes, metadata.duration)
        logger.info(f"[5/5] Clip extraction complete: {len(clips)} clips")
        
        # Associate keyframes with clips
        for clip in clips:
            clip.keyframes = [
                f.frame_id for f in frames
                if clip.start <= f.timestamp < clip.end
            ]
        
        # Save metadata
        result = {
            "metadata": asdict(metadata),
            "video_path": str(video_path.relative_to(output_dir)),
            "audio_path": str(audio_path.relative_to(output_dir)),
            "scenes": scenes,
            "frames": [asdict(f) for f in frames],
            "clips": [asdict(c) for c in clips]
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processing complete. Output: {output_dir}")
        return result


def process_youtube_video(
    url: str,
    output_base: str = "data/videos",
    frame_fps: float = 1.0,
    use_scene_detect: bool = True,
    scene_threshold: float = 0.3,
    clip_length: float = 15.0,
    use_scene_clips: bool = True,
    min_clip_length: float = 5.0,
    max_clip_length: float = 30.0
) -> dict:
    """
    Convenience function to process a YouTube video.
    
    Args:
        url: YouTube URL
        output_base: Base directory for outputs
        frame_fps: Frames per second for sampling
        use_scene_detect: Whether to use scene detection (via ffmpeg)
        scene_threshold: Scene detection threshold (0.0-1.0)
        clip_length: Fixed clip length (fallback)
        use_scene_clips: Whether to use scene-based clips
        min_clip_length: Minimum clip length
        max_clip_length: Maximum clip length
        
    Returns:
        Dictionary with all metadata and file paths
    """
    downloader = VideoDownloader(
        output_base=output_base,
        frame_fps=frame_fps,
        use_scene_detect=use_scene_detect,
        scene_threshold=scene_threshold,
        clip_length=clip_length,
        use_scene_clips=use_scene_clips,
        min_clip_length=min_clip_length,
        max_clip_length=max_clip_length
    )
    return downloader.process_video(url)


def _get_video_metadata_ffprobe(video_path: Path) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr}")
    return json.loads(result.stdout)


def _build_scene_based_clip_boundaries(
    scenes: List[Tuple[float, float]],
    clip_length: float,
    min_clip_length: float,
    max_clip_length: float,
) -> List[Tuple[float, float]]:
    """
    Accumulate contiguous scenes into valid clip boundaries.

    This avoids dropping videos with many short scenes by merging adjacent
    scenes until the clip reaches the minimum length, while still splitting
    clips that exceed the configured maximum length.
    """
    clip_boundaries: List[Tuple[float, float]] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    for start, end in scenes:
        if end <= start:
            continue

        if current_start is None:
            current_start = start
            current_end = end
            continue

        proposed_end = end
        proposed_duration = proposed_end - current_start

        if proposed_duration <= max_clip_length:
            current_end = proposed_end
            continue

        current_duration = current_end - current_start
        if current_duration >= min_clip_length:
            clip_boundaries.append((current_start, current_end))
            current_start = start
            current_end = end
            continue

        # The accumulated clip is still too short, so keep extending until it
        # becomes valid rather than dropping it.
        current_end = proposed_end
        if current_end - current_start >= min_clip_length:
            clip_boundaries.append((current_start, current_end))
            current_start = None
            current_end = None

    if current_start is not None and current_end is not None:
        current_duration = current_end - current_start
        if current_duration >= min_clip_length:
            clip_boundaries.append((current_start, current_end))
        elif clip_boundaries:
            prev_start, prev_end = clip_boundaries.pop()
            merged_end = current_end
            if merged_end - prev_start <= max_clip_length:
                clip_boundaries.append((prev_start, merged_end))
            else:
                clip_boundaries.append((prev_start, prev_end))
        else:
            clip_boundaries.append((current_start, current_end))

    # Split any remaining long clips into fixed-size chunks.
    normalized_boundaries: List[Tuple[float, float]] = []
    for start, end in clip_boundaries:
        duration = end - start
        if duration <= max_clip_length:
            normalized_boundaries.append((start, end))
            continue

        t = start
        while t < end:
            chunk_end = min(t + clip_length, end)
            if chunk_end - t >= min_clip_length:
                normalized_boundaries.append((t, chunk_end))
            elif normalized_boundaries:
                prev_start, prev_end = normalized_boundaries.pop()
                normalized_boundaries.append((prev_start, chunk_end))
            else:
                normalized_boundaries.append((t, chunk_end))
            t = chunk_end

    return normalized_boundaries


def process_local_video(
    video_path: str,
    output_dir: str,
    video_id: Optional[str] = None,
    frame_fps: float = 1.0,
    use_scene_detect: bool = True,
    scene_threshold: float = 0.3,
    clip_length: float = 15.0,
    use_scene_clips: bool = True,
    min_clip_length: float = 5.0,
    max_clip_length: float = 30.0
) -> dict:
    """
    Process a local video file (no YouTube download).

    Runs the same extraction pipeline as process_youtube_video but
    starting from a local .mp4 file: audio extraction, scene detection,
    frame extraction, and clip extraction.

    Args:
        video_path: Path to the local video file
        output_dir: Directory to write all outputs
        video_id: Optional video ID (defaults to filename stem)
        frame_fps: Frames per second for sampling
        use_scene_detect: Whether to use scene detection
        scene_threshold: Scene detection threshold (0.0-1.0)
        clip_length: Fixed clip length (fallback)
        use_scene_clips: Whether to use scene-based clips
        min_clip_length: Minimum clip length
        max_clip_length: Maximum clip length

    Returns:
        Dictionary with all metadata and file paths
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if video_id is None:
        video_id = video_path.stem

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frames").mkdir(exist_ok=True)
    (output_dir / "clips").mkdir(exist_ok=True)

    # Get metadata via ffprobe
    probe = _get_video_metadata_ffprobe(video_path)
    fmt = probe.get("format", {})
    duration = float(fmt.get("duration", 0))

    # Find video stream for dimensions/fps
    width, height, fps = 0, 0, 30.0
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))
            # Parse fps from r_frame_rate (e.g. "30/1")
            r_fps = stream.get("r_frame_rate", "30/1")
            if "/" in r_fps:
                num, den = r_fps.split("/")
                fps = float(num) / float(den) if float(den) > 0 else 30.0
            else:
                fps = float(r_fps)
            break

    metadata = VideoMetadata(
        video_id=video_id,
        title=video_path.stem,
        description="",
        duration=duration,
        upload_date="",
        channel="",
        url=str(video_path),
        download_time=datetime.now().isoformat(),
        width=width,
        height=height,
        fps=fps
    )

    logger.info(f"Processing local video: {video_id} ({duration:.1f}s, {width}x{height})")

    # Reuse VideoDownloader for extraction steps
    downloader = VideoDownloader(
        output_base=str(output_dir.parent),
        frame_fps=frame_fps,
        use_scene_detect=use_scene_detect,
        scene_threshold=scene_threshold,
        clip_length=clip_length,
        use_scene_clips=use_scene_clips,
        min_clip_length=min_clip_length,
        max_clip_length=max_clip_length
    )

    # Extract audio directly from the original video into the output dir
    logger.info("[1/4] Extracting audio...")
    # Check if video has an audio stream
    has_audio = any(
        s.get("codec_type") == "audio"
        for s in probe.get("streams", [])
    )

    audio_path = output_dir / "audio.wav"
    if has_audio:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"[1/4] Audio extraction failed (non-fatal): {result.stderr[:200]}")
        else:
            logger.info(f"[1/4] Audio extracted: {audio_path}")
    else:
        logger.info("[1/4] No audio stream found, skipping audio extraction")

    # Detect scenes from the original video
    logger.info("[2/4] Detecting scenes...")
    scenes = downloader.detect_scenes(video_path)
    logger.info(f"[2/4] Detected {len(scenes)} scenes")

    # Extract frames - ffmpeg writes to output_dir/frames/
    logger.info("[3/4] Extracting frames...")
    frames_dir = output_dir / "frames"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={frame_fps}",
        "-frame_pts", "1",
        str(frames_dir / "frame_%06d.jpg")
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr}")

    frame_interval = 1.0 / frame_fps
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    scene_starts = {s[0] for s in scenes} if scenes else set()

    frames = []
    keyframe_count = 0
    for i, frame_file in enumerate(frame_files):
        timestamp = i * frame_interval
        if timestamp > duration:
            break
        frame_id = f"frame_{i:06d}"
        is_keyframe = any(abs(timestamp - start) < frame_interval for start in scene_starts)
        if is_keyframe: keyframe_count += 1
        scene_id = None
        for j, (start, end) in enumerate(scenes):
            if start <= timestamp < end:
                scene_id = j
                break
        frames.append(FrameInfo(
            frame_id=frame_id,
            path=str(frame_file.relative_to(output_dir)),
            timestamp=timestamp,
            is_keyframe=is_keyframe,
            scene_id=scene_id
        ))
    logger.info(f"[3/4] Extracted {len(frames)} frames, and ({keyframe_count} are keyframes)")

    # Extract clips from the original video into output_dir/clips/
    logger.info("[4/4] Extracting clips...")
    clips_dir = output_dir / "clips"
    clips = []

    # Determine clip boundaries for local processing.
    if downloader.use_scene_clips and scenes:
        clip_boundaries = _build_scene_based_clip_boundaries(
            scenes=scenes,
            clip_length=downloader.clip_length,
            min_clip_length=downloader.min_clip_length,
            max_clip_length=downloader.max_clip_length,
        )
        if not clip_boundaries:
            logger.warning(
                "[4/4] Scene-based clip construction produced 0 clips; "
                "falling back to fixed-length clips"
            )
    else:
        clip_boundaries = []
    if not clip_boundaries:
        for t in range(0, int(duration), int(downloader.clip_length)):
            start = float(t)
            end = min(start + downloader.clip_length, duration)
            if end - start >= downloader.min_clip_length:
                clip_boundaries.append((start, end))
        if not clip_boundaries and duration > 0:
            clip_boundaries.append((0.0, duration))

    for i, (start, end) in enumerate(clip_boundaries):
        clip_id = f"clip_{i:04d}"
        clip_path = clips_dir / f"{clip_id}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(end - start),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "ultrafast",
            str(clip_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Failed to extract clip {clip_id}: {result.stderr}")
            continue
        clips.append(ClipInfo(
            clip_id=clip_id,
            path=str(clip_path.relative_to(output_dir)),
            start=start,
            end=end,
            duration=end - start,
            scene_id=i if scenes else None,
            keyframes=[]
        ))
    logger.info(f"[4/4] Extracted {len(clips)} clips")

    # Associate keyframes with clips
    for clip in clips:
        clip.keyframes = [
            f.frame_id for f in frames
            if clip.start <= f.timestamp < clip.end
        ]

    # Save metadata
    result = {
        "metadata": asdict(metadata),
        "video_path": str(video_path),  # original source path (not copied)
        "audio_path": "audio.wav",
        "scenes": scenes,
        "frames": [asdict(f) for f in frames],
        "clips": [asdict(c) for c in clips]
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Local video processing complete. Output: {output_dir}")
    return result

