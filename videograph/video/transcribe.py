"""
Transcription module using OpenAI Whisper API.

Produces transcript with word-level or segment-level timestamps.
Handles large audio files by chunking.
"""

import json
import logging
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
from openai import OpenAI

from ..cache.openai_cache import get_cache

logger = logging.getLogger(__name__)

# OpenAI Whisper API file size limit (25MB)
MAX_FILE_SIZE_MB = 24  # Leave some margin


@dataclass
class TranscriptSegment:
    """A segment of transcript with timestamps."""
    id: int
    text: str
    start: float  # seconds
    end: float  # seconds
    words: Optional[List[dict]] = None  # word-level timestamps if available


@dataclass
class Transcript:
    """Full transcript with segments."""
    segments: List[TranscriptSegment]
    full_text: str
    language: str
    duration: float


class Transcriber:
    """Transcribes audio using OpenAI Whisper API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-1",
        language: Optional[str] = None,
        timestamp_granularity: str = "segment",
        cache_enabled: bool = True
    ):
        """
        Initialize the transcriber.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Whisper model name
            language: Language code (None for auto-detect)
            timestamp_granularity: "word" or "segment"
            cache_enabled: Whether to cache API calls
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.language = language
        self.timestamp_granularity = timestamp_granularity
        self.cache = get_cache() if cache_enabled else None
    
    def transcribe(self, audio_path: Path) -> Transcript:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file (WAV, MP3, etc.)
            
        Returns:
            Transcript object with segments and timestamps
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing {audio_path}...")
        
        # Prepare API call parameters
        params = {
            "model": self.model,
            "response_format": "verbose_json",
            "timestamp_granularities": [self.timestamp_granularity]
        }
        if self.language:
            params["language"] = self.language

        # Check file size and compress if needed
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.info(f"Audio file too large ({file_size_mb:.1f}MB), compressing...")
            audio_path = self._compress_audio(audio_path)
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)
            logger.info(f"Compressed to {file_size_mb:.1f}MB")

        # If still too large, split into chunks and merge transcripts.
        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.info(
                f"Compressed audio still above limit ({file_size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB), "
                "splitting into chunks for transcription..."
            )
            return self._transcribe_chunked(audio_path, params)

        response_dict = self._transcribe_single_file(audio_path, params)
        return self._parse_response(response_dict)

    def _transcribe_single_file(self, audio_path: Path, params: dict) -> dict:
        """Transcribe one audio file with cache support and return response dict."""
        cache_key = {
            "file_hash": self._hash_file(audio_path),
            "params": params
        }
        
        if self.cache:
            cached = self.cache.get(
                self.model,
                [{"role": "audio", "content": str(cache_key)}],
                params
            )
            if cached:
                logger.info(f"Using cached transcription for {audio_path.name}")
                return cached

        with open(audio_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                file=audio_file,
                **params
            )

        response_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        
        # Cache the response
        if self.cache:
            self.cache.set(
                self.model,
                [{"role": "audio", "content": str(cache_key)}],
                params,
                response_dict
            )

        return response_dict

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds using ffprobe."""
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"ffprobe failed for {audio_path}")
        return float(result.stdout.strip())

    def _split_audio_into_chunks(
        self,
        audio_path: Path,
        output_dir: Path,
        target_chunk_mb: float = 18.0
    ) -> List[dict]:
        """
        Split audio into size-safe chunks using duration-based partitioning.

        Returns a list of dicts:
            [{"path": Path(...), "start": float, "end": float}, ...]
        """
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        duration = self._get_audio_duration(audio_path)

        # Add a small safety factor so chunks land comfortably under the limit.
        chunk_count = max(2, math.ceil((file_size_mb / target_chunk_mb) * 1.1))
        chunk_duration = duration / chunk_count

        logger.info(
            f"Splitting {audio_path.name} ({file_size_mb:.1f}MB, {duration:.1f}s) "
            f"into {chunk_count} chunks (~{chunk_duration:.1f}s each)"
        )

        chunks = []
        for i in range(chunk_count):
            start = i * chunk_duration
            end = duration if i == chunk_count - 1 else min((i + 1) * chunk_duration, duration)
            if end <= start:
                continue

            chunk_path = output_dir / f"chunk_{i:03d}.mp3"
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", str(audio_path),
                "-t", str(end - start),
                "-ac", "1",
                "-ar", "16000",
                "-b:a", "64k",
                str(chunk_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create chunk {i}: {result.stderr}")

            chunk_size_mb = chunk_path.stat().st_size / (1024 * 1024)
            if chunk_size_mb > MAX_FILE_SIZE_MB:
                # One more compression pass for an oversized chunk.
                logger.info(
                    f"Chunk {i} still large ({chunk_size_mb:.1f}MB), compressing chunk..."
                )
                compressed_chunk = self._compress_audio(chunk_path)
                chunk_path = compressed_chunk
                chunk_size_mb = chunk_path.stat().st_size / (1024 * 1024)
                if chunk_size_mb > MAX_FILE_SIZE_MB:
                    raise RuntimeError(
                        f"Chunk {i} still above limit after compression ({chunk_size_mb:.1f}MB)"
                    )

            chunks.append({"path": chunk_path, "start": start, "end": end})

        return chunks

    def _offset_segment_times(self, segments: List[dict], offset_s: float) -> List[dict]:
        """Apply a timestamp offset to segment and word timestamps."""
        adjusted = []
        for seg in segments:
            seg_copy = dict(seg)
            seg_copy["start"] = float(seg_copy.get("start", 0.0)) + offset_s
            seg_copy["end"] = float(seg_copy.get("end", 0.0)) + offset_s

            words = seg_copy.get("words")
            if isinstance(words, list):
                adjusted_words = []
                for w in words:
                    if isinstance(w, dict):
                        w_copy = dict(w)
                        if "start" in w_copy:
                            w_copy["start"] = float(w_copy["start"]) + offset_s
                        if "end" in w_copy:
                            w_copy["end"] = float(w_copy["end"]) + offset_s
                        adjusted_words.append(w_copy)
                    else:
                        adjusted_words.append(w)
                seg_copy["words"] = adjusted_words

            adjusted.append(seg_copy)
        return adjusted

    def _transcribe_chunked(self, audio_path: Path, params: dict) -> Transcript:
        """Split oversized audio, transcribe chunks, then merge with adjusted timestamps."""
        duration = self._get_audio_duration(audio_path)

        merged_segments = []
        merged_text_parts = []
        language = None

        with tempfile.TemporaryDirectory(prefix=f"{audio_path.stem}_chunks_", dir=str(audio_path.parent)) as tmp_dir:
            chunk_dir = Path(tmp_dir)
            chunks = self._split_audio_into_chunks(audio_path, chunk_dir)

            for idx, chunk in enumerate(chunks, start=1):
                chunk_path = chunk["path"]
                start = chunk["start"]
                logger.info(
                    f"Transcribing chunk {idx}/{len(chunks)} "
                    f"({chunk_path.name}, start={start:.1f}s)"
                )

                response_dict = self._transcribe_single_file(chunk_path, params)
                chunk_segments = response_dict.get("segments", [])
                if isinstance(chunk_segments, list):
                    merged_segments.extend(self._offset_segment_times(chunk_segments, start))

                text = str(response_dict.get("text", "") or "").strip()
                if text:
                    merged_text_parts.append(text)

                if not language:
                    lang = response_dict.get("language")
                    if isinstance(lang, str) and lang.strip():
                        language = lang.strip()

        merged_segments.sort(key=lambda s: float(s.get("start", 0.0)))

        merged_response = {
            "text": " ".join(merged_text_parts).strip(),
            "language": language or "unknown",
            "duration": duration,
            "segments": merged_segments,
        }
        logger.info(f"Merged transcript from {len(merged_text_parts)} chunks")
        return self._parse_response(merged_response)
    
    def _compress_audio(self, audio_path: Path) -> Path:
        """Compress audio file to fit within API limits."""
        # Create compressed file path
        compressed_path = audio_path.parent / f"{audio_path.stem}_compressed.mp3"
        
        # Use ffmpeg to compress to MP3 with lower bitrate
        # Target ~20MB for safety margin
        target_size_kb = (MAX_FILE_SIZE_MB - 4) * 1024  # Leave 4MB margin
        
        # Get duration to calculate bitrate
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        duration = float(result.stdout.strip()) if result.returncode == 0 else 600
        
        # Calculate target bitrate (kbps)
        target_bitrate = int((target_size_kb * 8) / duration)
        target_bitrate = max(32, min(target_bitrate, 128))  # Clamp between 32-128 kbps
        
        logger.info(f"Compressing audio to {target_bitrate}kbps...")
        
        compress_cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ac", "1",  # Mono
            "-ar", "16000",  # 16kHz sample rate (good for speech)
            "-b:a", f"{target_bitrate}k",
            str(compressed_path)
        ]
        
        result = subprocess.run(compress_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Compression failed: {result.stderr}")
            raise RuntimeError(f"Audio compression failed: {result.stderr}")
        
        return compressed_path
    
    def _hash_file(self, path: Path) -> str:
        """Compute a hash of the file for caching."""
        import hashlib
        hasher = hashlib.md5()
        with open(path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _parse_response(self, response: dict) -> Transcript:
        """Parse Whisper API response into Transcript object."""
        segments = []
        
        for i, seg in enumerate(response.get("segments", [])):
            segment = TranscriptSegment(
                id=i,
                text=seg.get("text", "").strip(),
                start=seg.get("start", 0),
                end=seg.get("end", 0),
                words=seg.get("words")
            )
            segments.append(segment)
        
        # If no segments but we have text, create a single segment
        if not segments and response.get("text"):
            segments.append(TranscriptSegment(
                id=0,
                text=response["text"].strip(),
                start=0,
                end=response.get("duration", 0),
                words=None
            ))
        
        return Transcript(
            segments=segments,
            full_text=response.get("text", "").strip(),
            language=response.get("language", "unknown"),
            duration=response.get("duration", 0)
        )
    
    def save_transcript(self, transcript: Transcript, output_path: Path):
        """Save transcript to JSON file."""
        output_path = Path(output_path)
        
        data = {
            "full_text": transcript.full_text,
            "language": transcript.language,
            "duration": transcript.duration,
            "segments": [asdict(s) for s in transcript.segments]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Transcript saved to {output_path}")
    
    def create_sentence_segments(self, transcript: Transcript) -> List[TranscriptSegment]:
        """
        Split transcript into sentence-level segments with aligned timestamps.
        
        This interpolates timestamps for sentence boundaries within segments.
        """
        import re
        
        sentence_segments = []
        sentence_id = 0
        
        for segment in transcript.segments:
            text = segment.text
            
            # Split by sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            if len(sentences) == 1:
                # Single sentence, use original timestamps
                sentence_segments.append(TranscriptSegment(
                    id=sentence_id,
                    text=text.strip(),
                    start=segment.start,
                    end=segment.end,
                    words=segment.words
                ))
                sentence_id += 1
            else:
                # Multiple sentences, interpolate timestamps
                segment_duration = segment.end - segment.start
                total_chars = sum(len(s) for s in sentences)
                
                current_time = segment.start
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    # Proportional duration based on character length
                    proportion = len(sentence) / max(total_chars, 1)
                    duration = segment_duration * proportion
                    
                    sentence_segments.append(TranscriptSegment(
                        id=sentence_id,
                        text=sentence.strip(),
                        start=current_time,
                        end=current_time + duration,
                        words=None
                    ))
                    
                    current_time += duration
                    sentence_id += 1
        
        return sentence_segments


def transcribe_audio(
    audio_path: str,
    output_dir: Optional[str] = None,
    model: str = "whisper-1",
    language: Optional[str] = None,
    timestamp_granularity: str = "segment"
) -> dict:
    """
    Convenience function to transcribe audio and save results.
    
    Args:
        audio_path: Path to audio file
        output_dir: Directory to save transcript (defaults to same as audio)
        model: Whisper model name
        language: Language code (None for auto-detect)
        timestamp_granularity: "word" or "segment"
        
    Returns:
        Dictionary with transcript data
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir) if output_dir else audio_path.parent
    
    transcriber = Transcriber(
        model=model,
        language=language,
        timestamp_granularity=timestamp_granularity
    )
    
    transcript = transcriber.transcribe(audio_path)
    
    # Create sentence-level segments
    sentence_segments = transcriber.create_sentence_segments(transcript)
    
    # Save main transcript
    transcript_path = output_dir / "transcript.json"
    transcriber.save_transcript(transcript, transcript_path)
    
    # Save sentence segments
    sentences_data = {
        "sentence_segments": [asdict(s) for s in sentence_segments]
    }
    sentences_path = output_dir / "transcript_sentences.json"
    with open(sentences_path, 'w', encoding='utf-8') as f:
        json.dump(sentences_data, f, indent=2, ensure_ascii=False)
    
    return {
        "transcript": asdict(transcript) if hasattr(transcript, '__dataclass_fields__') else {
            "full_text": transcript.full_text,
            "language": transcript.language,
            "duration": transcript.duration,
            "segments": [asdict(s) for s in transcript.segments]
        },
        "sentence_segments": [asdict(s) for s in sentence_segments],
        "transcript_path": str(transcript_path),
        "sentences_path": str(sentences_path)
    }


