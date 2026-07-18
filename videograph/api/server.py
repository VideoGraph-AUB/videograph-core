"""
FastAPI backend for VideoGraph.

Endpoints:
- POST /api/videos/process - Process a YouTube video
- GET /api/videos/{video_id} - Get video metadata and graph
- POST /api/videos/{video_id}/query - Query the graph
- GET /api/videos/{video_id}/node/{node_id} - Get node details
- GET /api/config - Get current configuration
- POST /api/config - Update configuration
"""

import asyncio
import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VideoGraph",
    description="Multimodal video understanding via graph-based reasoning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Will be configured from config
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "videos"
CONFIG_DIR = BASE_DIR / "config"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ============ Pydantic Models ============

class ProcessVideoRequest(BaseModel):
    youtube_url: str


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    hop_expansion: Optional[int] = None


class ConfigUpdate(BaseModel):
    text_model: Optional[str] = None
    vision_model: Optional[str] = None
    temperature: Optional[float] = None
    cache_enabled: Optional[bool] = None


class ProcessingStatus(BaseModel):
    video_id: str
    status: str  # pending, processing, completed, failed
    stage: str
    progress: float
    message: Optional[str] = None
    sub_stage: Optional[str] = None
    sub_progress: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    estimated_remaining: Optional[float] = None


# ============ Global State ============

processing_status: Dict[str, ProcessingStatus] = {}
processing_start_times: Dict[str, float] = {}


# ============ Helper Functions ============

def load_config() -> dict:
    """Load configuration from default.yaml and apply saved UI overrides."""
    config_path = CONFIG_DIR / "default.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    ui_state = load_ui_state()
    openai_config = config.setdefault("openai", {})
    cache_config = config.setdefault("cache", {})

    if "selected_text_model" in ui_state:
        openai_config["text_model"] = ui_state["selected_text_model"]
    if "selected_vision_model" in ui_state:
        openai_config["vision_model"] = ui_state["selected_vision_model"]
    if "temperature" in ui_state:
        openai_config["temperature"] = ui_state["temperature"]
    if "cache_enabled" in ui_state:
        cache_config["enabled"] = ui_state["cache_enabled"]

    return config


def load_ui_state() -> dict:
    """Load UI state from ui_state.json."""
    state_path = CONFIG_DIR / "ui_state.json"
    if state_path.exists():
        with open(state_path, 'r') as f:
            return json.load(f)
    return {}


def save_ui_state(state: dict):
    """Save UI state to ui_state.json."""
    state_path = CONFIG_DIR / "ui_state.json"
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)


def get_video_dir(video_id: str) -> Path:
    """Get the directory for a video."""
    return DATA_DIR / video_id


def log_event(video_id: str, event: str, data: dict = None):
    """Log an event to the video's log file."""
    log_path = LOGS_DIR / f"{video_id}.jsonl"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        "data": data or {}
    }
    with open(log_path, 'a') as f:
        f.write(json.dumps(entry) + "\n")


# ============ Background Task: Process Video ============

def create_progress_callback(video_id: str, stage: str, base_progress: float, stage_weight: float):
    """Create a progress callback for a processing stage."""
    import time
    
    def callback(progress):
        # progress is a ProcessingProgress object from parallel.py
        sub_progress = progress.current / max(progress.total, 1)
        total_progress = base_progress + (sub_progress * stage_weight)
        
        elapsed = time.time() - processing_start_times.get(video_id, time.time())
        
        processing_status[video_id] = ProcessingStatus(
            video_id=video_id,
            status="processing",
            stage=stage,
            progress=total_progress,
            message=progress.message,
            sub_stage=progress.stage,
            sub_progress=sub_progress,
            elapsed_seconds=elapsed,
            estimated_remaining=progress.elapsed_seconds * (progress.total - progress.current) / max(progress.current, 1) if progress.current > 0 else None
        )
    
    return callback


async def process_video_task(video_id: str, youtube_url: str, config: dict):
    """Background task to process a video with parallel processing."""
    import time
    from ..video.io import VideoDownloader
    from ..video.adaptive_ingest import process_local_video_adaptive
    from ..video.transcribe import transcribe_audio
    from ..visual.adaptive_processing import (
        analyze_adaptive_clips,
        update_adaptive_visual_json_with_ocr,
    )
    from ..graph.builder import build_video_graph
    from ..cache.openai_cache import get_cache
    
    try:
        video_dir = get_video_dir(video_id)
        processing_start_times[video_id] = time.time()
        get_cache().enabled = bool(config.get("cache", {}).get("enabled", True))
        
        logger.info(f"=" * 60)
        logger.info(f"PROCESSING VIDEO: {video_id}")
        logger.info(f"URL: {youtube_url}")
        logger.info(f"Output: {video_dir}")
        logger.info(f"=" * 60)
        
        # Stage weights (must sum to 1.0)
        # ingest: 25%, transcribe: 10%, visual: 30%, ocr: 10%, graph: 25%

        # Stage 1: Download and adaptive preprocessing (0.0 - 0.25)
        logger.info("[STAGE 1/5] Starting video download and adaptive preprocessing...")
        processing_status[video_id] = ProcessingStatus(
            video_id=video_id,
            status="processing",
            stage="ingest",
            progress=0.0,
            message="Downloading and segmenting video...",
            elapsed_seconds=0
        )
        log_event(video_id, "stage_start", {"stage": "ingest"})

        downloader = VideoDownloader(output_base=str(DATA_DIR))
        video_path, download_metadata = downloader.download_video(youtube_url)
        process_local_video_adaptive(
            video_path=str(video_path),
            output_dir=str(video_dir),
            video_id=video_id,
            config=config,
        )
        metadata_path = video_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            metadata.setdefault("metadata", {}).update({
                "title": download_metadata.title,
                "description": download_metadata.description,
                "upload_date": download_metadata.upload_date,
                "channel": download_metadata.channel,
                "url": download_metadata.url,
            })
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        log_event(video_id, "stage_complete", {"stage": "ingest"})
        logger.info("[STAGE 1/5] Video download and adaptive preprocessing complete!")

        # Stage 2: Transcribe audio (0.25 - 0.35)
        logger.info("[STAGE 2/5] Starting audio transcription...")
        elapsed = time.time() - processing_start_times[video_id]
        processing_status[video_id] = ProcessingStatus(
            video_id=video_id,
            status="processing",
            stage="transcribe",
            progress=0.25,
            message="Transcribing audio...",
            elapsed_seconds=elapsed
        )
        log_event(video_id, "stage_start", {"stage": "transcribe"})
        
        audio_path = video_dir / "audio.wav"
        trans_config = config.get("transcription", {})
        transcribe_audio(
            str(audio_path),
            output_dir=str(video_dir),
            model=config.get("openai", {}).get("transcription_model", "whisper-1"),
            language=trans_config.get("language"),
            timestamp_granularity=trans_config.get("timestamp_granularity", "segment")
        )
        log_event(video_id, "stage_complete", {"stage": "transcribe"})
        logger.info("[STAGE 2/5] Transcription complete!")
        
        # Stage 3: Adaptive visual captioning with parallel processing (0.35 - 0.65)
        logger.info("[STAGE 3/5] Starting adaptive visual captioning...")
        elapsed = time.time() - processing_start_times[video_id]
        processing_status[video_id] = ProcessingStatus(
            video_id=video_id,
            status="processing",
            stage="visual_captioning",
            progress=0.35,
            message="Analyzing adaptive clip keyframes...",
            elapsed_seconds=elapsed
        )
        log_event(video_id, "stage_start", {"stage": "visual_captioning"})
        
        visual_config = config.get("visual", {})
        max_parallel = config.get("processing", {}).get("max_parallel_vision", 5)
        
        analyze_adaptive_clips(
            str(video_dir),
            model=config.get("openai", {}).get("vision_model", "gpt-4o"),
            prompt_style=visual_config.get("vision_prompt_style", "detailed"),
            temperature=float(config.get("openai", {}).get("temperature", 0.0)),
            max_parallel=max_parallel,
            append_state_change_to_description=bool(
                visual_config.get("append_state_change_to_description", False)
            ),
            progress_callback=create_progress_callback(video_id, "visual_captioning", 0.35, 0.30)
        )
        log_event(video_id, "stage_complete", {"stage": "visual_captioning"})
        logger.info("[STAGE 3/5] Visual captioning complete!")

        # Stage 4: OCR with parallel processing (0.65 - 0.75)
        logger.info("[STAGE 4/5] Starting OCR extraction...")
        elapsed = time.time() - processing_start_times[video_id]
        processing_status[video_id] = ProcessingStatus(
            video_id=video_id,
            status="processing",
            stage="ocr",
            progress=0.65,
            message="Extracting on-screen text...",
            elapsed_seconds=elapsed
        )
        log_event(video_id, "stage_start", {"stage": "ocr"})
        
        if visual_config.get("ocr_enabled", True):
            try:
                update_adaptive_visual_json_with_ocr(
                    str(video_dir),
                    model=config.get("openai", {}).get("vision_model", "gpt-4o"),
                    max_parallel=max_parallel,
                    progress_callback=create_progress_callback(video_id, "ocr", 0.65, 0.10)
                )
            except Exception as e:
                logger.warning(f"OCR failed: {e}")
        log_event(video_id, "stage_complete", {"stage": "ocr"})
        logger.info("[STAGE 4/5] OCR complete!")
        
        # Stage 5: Build graph + embeddings (0.75 - 1.0)
        logger.info("[STAGE 5/5] Building knowledge graph + computing embeddings...")
        elapsed = time.time() - processing_start_times[video_id]
        processing_status[video_id] = ProcessingStatus(
            video_id=video_id,
            status="processing",
            stage="graph_build",
            progress=0.75,
            message="Building knowledge graph...",
            elapsed_seconds=elapsed
        )
        log_event(video_id, "stage_start", {"stage": "graph_build"})
        
        max_parallel_embeddings = config.get("processing", {}).get("max_parallel_embeddings", 10)
        
        build_video_graph(
            str(video_dir),
            compute_embeddings=True,
            max_parallel_embeddings=max_parallel_embeddings,
            progress_callback=create_progress_callback(video_id, "graph_build", 0.75, 0.25),
            config=config,
        )
        log_event(video_id, "stage_complete", {"stage": "graph_build"})
        logger.info("[STAGE 5/5] Graph build complete!")
        
        # Complete
        total_elapsed = time.time() - processing_start_times[video_id]
        logger.info("=" * 60)
        logger.info(f"VIDEO PROCESSING COMPLETE: {video_id}")
        logger.info(f"Total time: {total_elapsed:.1f}s")
        logger.info("=" * 60)
        processing_status[video_id] = ProcessingStatus(
            video_id=video_id,
            status="completed",
            stage="done",
            progress=1.0,
            message="Processing complete!",
            elapsed_seconds=total_elapsed
        )
        log_event(video_id, "processing_complete", {"total_seconds": total_elapsed})
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")
        import traceback
        traceback.print_exc()
        elapsed = time.time() - processing_start_times.get(video_id, time.time())
        processing_status[video_id] = ProcessingStatus(
            video_id=video_id,
            status="failed",
            stage="error",
            progress=0,
            message=str(e),
            elapsed_seconds=elapsed
        )
        log_event(video_id, "error", {"error": str(e)})


# ============ API Endpoints ============

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    config = load_config()
    ui_state = load_ui_state()
    
    # Merge with UI overrides
    merged = {
        "openai": {
            "text_model": ui_state.get("selected_text_model", config.get("openai", {}).get("text_model", "gpt-4o")),
            "vision_model": ui_state.get("selected_vision_model", config.get("openai", {}).get("vision_model", "gpt-4o")),
            "temperature": ui_state.get("temperature", config.get("openai", {}).get("temperature", 0.3)),
        },
        "cache_enabled": ui_state.get("cache_enabled", config.get("cache", {}).get("enabled", True)),
        "available_models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo"
        ]
    }
    
    return merged


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    """Update configuration."""
    ui_state = load_ui_state()
    
    if update.text_model:
        ui_state["selected_text_model"] = update.text_model
    if update.vision_model:
        ui_state["selected_vision_model"] = update.vision_model
    if update.temperature is not None:
        ui_state["temperature"] = update.temperature
    if update.cache_enabled is not None:
        ui_state["cache_enabled"] = update.cache_enabled
    
    save_ui_state(ui_state)
    return {"status": "updated", "config": ui_state}


@app.get("/api/videos")
async def list_videos():
    """List all processed videos."""
    videos = []
    
    for video_dir in DATA_DIR.iterdir():
        if video_dir.is_dir():
            metadata_path = video_dir / "metadata.json"
            graph_path = video_dir / "graph.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                videos.append({
                    "video_id": video_dir.name,
                    "title": metadata.get("metadata", {}).get("title", ""),
                    "duration": metadata.get("metadata", {}).get("duration", 0),
                    "has_graph": graph_path.exists(),
                    "status": processing_status.get(video_dir.name, ProcessingStatus(
                        video_id=video_dir.name,
                        status="completed" if graph_path.exists() else "pending",
                        stage="done" if graph_path.exists() else "unknown",
                        progress=1.0 if graph_path.exists() else 0
                    )).status
                })
    
    return {"videos": videos}


@app.post("/api/videos/process")
async def process_video(request: ProcessVideoRequest):
    """Start processing a YouTube video."""
    from ..video.io import extract_video_id
    
    try:
        video_id = extract_video_id(request.youtube_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Check if already processed
    video_dir = get_video_dir(video_id)
    if video_dir.exists() and (video_dir / "graph.json").exists():
        return {
            "video_id": video_id,
            "status": "already_processed",
            "message": "Video already processed. Use DELETE to reprocess."
        }
    
    # Initialize status
    processing_status[video_id] = ProcessingStatus(
        video_id=video_id,
        status="pending",
        stage="queued",
        progress=0,
        message="Queued for processing"
    )
    
    # Load config
    config = load_config()
    
    # Start processing in a real background thread
    def _run_task():
        asyncio.run(process_video_task(video_id, request.youtube_url, config))

    thread = threading.Thread(target=_run_task, daemon=True)
    thread.start()
    
    return {
        "video_id": video_id,
        "status": "processing",
        "message": "Video processing started"
    }


@app.get("/api/videos/{video_id}/status")
async def get_video_status(video_id: str):
    """Get processing status for a video."""
    if video_id in processing_status:
        return processing_status[video_id]
    
    video_dir = get_video_dir(video_id)
    if video_dir.exists() and (video_dir / "graph.json").exists():
        return ProcessingStatus(
            video_id=video_id,
            status="completed",
            stage="done",
            progress=1.0
        )
    
    raise HTTPException(status_code=404, detail="Video not found")


def sanitize_json_values(obj):
    """Replace Infinity/-Infinity with None for JSON compliance."""
    if isinstance(obj, dict):
        return {k: sanitize_json_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json_values(item) for item in obj]
    elif isinstance(obj, float):
        if obj == float('inf') or obj == float('-inf'):
            return None
        return obj
    return obj


def load_json_safe(path):
    """Load JSON file, handling Infinity values."""
    with open(path, 'r') as f:
        content = f.read()
    # Replace JavaScript Infinity with null for JSON parsing
    content = content.replace(': Infinity', ': null')
    content = content.replace(': -Infinity', ': null')
    return json.loads(content)


@app.get("/api/videos/{video_id}")
async def get_video(video_id: str):
    """Get video metadata, transcript, and graph."""
    video_dir = get_video_dir(video_id)
    
    if not video_dir.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    result = {"video_id": video_id}
    
    # Load metadata
    metadata_path = video_dir / "metadata.json"
    if metadata_path.exists():
        result["metadata"] = load_json_safe(metadata_path)
    
    # Load transcript
    transcript_path = video_dir / "transcript.json"
    if transcript_path.exists():
        result["transcript"] = load_json_safe(transcript_path)
    
    # Load graph
    graph_path = video_dir / "graph.json"
    if graph_path.exists():
        result["graph"] = load_json_safe(graph_path)
    
    # Load visual
    visual_path = video_dir / "visual.json"
    if visual_path.exists():
        result["visual"] = load_json_safe(visual_path)
    
    return result


@app.delete("/api/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a processed video."""
    video_dir = get_video_dir(video_id)
    
    if not video_dir.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    shutil.rmtree(video_dir)
    
    if video_id in processing_status:
        del processing_status[video_id]
    
    return {"status": "deleted", "video_id": video_id}


@app.post("/api/videos/{video_id}/query")
async def query_video(video_id: str, request: QueryRequest):
    """Query the video graph."""
    from ..qa.answer import answer_question
    
    video_dir = get_video_dir(video_id)
    graph_path = video_dir / "graph.json"
    
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail="Graph not found")
    
    config = load_config()
    openai_config = config.get("openai", {})
    retrieval_config = config.get("retrieval", {})
    text_model = openai_config.get("text_model", "gpt-4o")
    temperature = float(openai_config.get("temperature", 0.0))
    top_k = request.top_k if request.top_k is not None else int(retrieval_config.get("top_k", 10))
    hop_expansion = (
        request.hop_expansion
        if request.hop_expansion is not None
        else int(retrieval_config.get("hop_expansion", 2))
    )
    hybrid_alpha = float(retrieval_config.get("hybrid_alpha", 0.7))
    
    # Log query
    log_event(video_id, "query", {"query": request.query})
    
    result = answer_question(
        request.query,
        str(graph_path),
        text_model=text_model,
        temperature=temperature,
        top_k=top_k,
        hop_expansion=hop_expansion,
        hybrid_alpha=hybrid_alpha,
    )
    
    # Sanitize result to remove infinity values
    result = sanitize_json_values(result)
    
    log_event(video_id, "query_complete", {
        "answer_length": len(result["answer"]),
        "evidence_count": len(result["evidence"])
    })
    
    return result


@app.get("/api/videos/{video_id}/node/{node_id}")
async def get_node(video_id: str, node_id: str):
    """Get details for a specific node."""
    video_dir = get_video_dir(video_id)
    graph_path = video_dir / "graph.json"
    
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail="Graph not found")
    
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)
    
    # Find node
    for node in graph_data.get("nodes", []):
        if node.get("id") == node_id:
            # Get connected edges
            edges = [
                e for e in graph_data.get("edges", [])
                if e.get("source") == node_id or e.get("target") == node_id
            ]
            return {
                "node": node,
                "edges": edges
            }
    
    raise HTTPException(status_code=404, detail="Node not found")


@app.get("/api/videos/{video_id}/seek/{timestamp}")
async def seek_to_timestamp(video_id: str, timestamp: float):
    """Get nodes at a specific timestamp."""
    from ..retrieval.graph_retrieval import GraphRetriever
    from ..graph.serialization import load_graph_json
    
    video_dir = get_video_dir(video_id)
    graph_path = video_dir / "graph.json"
    
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail="Graph not found")
    
    graph = load_graph_json(graph_path)
    retriever = GraphRetriever()
    
    results = retriever.retrieve_by_timestamp(timestamp, graph, window=5.0)
    
    return {
        "timestamp": timestamp,
        "nodes": [
            {
                "node_id": r.node_id,
                "node_type": r.node_type,
                "text": r.text,
                "start": r.start,
                "end": r.end,
                "clip_id": r.clip_id
            }
            for r in results[:10]
        ]
    }


# Serve video files
@app.get("/api/videos/{video_id}/video")
async def get_video_file(video_id: str):
    """Serve the video file."""
    video_dir = get_video_dir(video_id)
    video_path = video_dir / "video.mp4"
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/api/videos/{video_id}/clips/{clip_id}")
async def get_clip_file(video_id: str, clip_id: str):
    """Serve a clip file."""
    video_dir = get_video_dir(video_id)
    clip_path = video_dir / "clips" / f"{clip_id}.mp4"
    
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    
    return FileResponse(clip_path, media_type="video/mp4")


@app.get("/api/videos/{video_id}/frames/{frame_id}")
async def get_frame_file(video_id: str, frame_id: str):
    """Serve a frame file."""
    video_dir = get_video_dir(video_id)
    frame_path = video_dir / "frames" / f"{frame_id}.jpg"
    
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    
    return FileResponse(frame_path, media_type="image/jpeg")


# ============ Global Search Across All Videos ============

class GlobalSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10


def keyword_search_video(video_dir: Path, query: str, video_title: str) -> List[dict]:
    """
    Fallback keyword search for videos without embeddings.
    Searches transcript and graph nodes for query terms.
    """
    import re
    results = []
    query_lower = query.lower()
    query_words = set(re.findall(r'\w+', query_lower))
    
    # Search transcript
    transcript_path = video_dir / "transcript.json"
    if transcript_path.exists():
        try:
            with open(transcript_path, 'r') as f:
                transcript = json.load(f)
            
            for segment in transcript.get('segments', []):
                text = segment.get('text', '')
                text_lower = text.lower()
                
                # Check for keyword matches
                text_words = set(re.findall(r'\w+', text_lower))
                overlap = len(query_words & text_words)
                
                if overlap > 0 or any(word in text_lower for word in query_words):
                    # Score based on overlap
                    score = overlap / max(len(query_words), 1)
                    # Boost for exact phrase match
                    if query_lower in text_lower:
                        score += 0.5
                    
                    results.append({
                        'video_id': video_dir.name,
                        'video_title': video_title,
                        'text': text[:500],
                        'start': segment.get('start', 0) or 0,
                        'score': min(score, 1.0)
                    })
        except Exception as e:
            logger.warning(f"Failed to search transcript {video_dir.name}: {e}")
    
    # Also search graph nodes (for visual descriptions, entities, etc.)
    graph_path = video_dir / "graph.json"
    if graph_path.exists():
        try:
            graph_data = load_json_safe(graph_path)
            
            for node in graph_data.get('nodes', []):
                text = node.get('text', '') or node.get('visual_description', '') or node.get('title', '') or node.get('name', '') or ''
                if not text:
                    continue
                    
                text_lower = text.lower()
                text_words = set(re.findall(r'\w+', text_lower))
                overlap = len(query_words & text_words)
                
                if overlap > 0 or any(word in text_lower for word in query_words):
                    score = overlap / max(len(query_words), 1)
                    if query_lower in text_lower:
                        score += 0.5
                    
                    results.append({
                        'video_id': video_dir.name,
                        'video_title': video_title,
                        'text': text[:500],
                        'start': node.get('start', 0) or 0,
                        'score': min(score, 1.0)
                    })
        except Exception as e:
            logger.warning(f"Failed to search graph {video_dir.name}: {e}")
    
    return results


@app.post("/api/search")
async def global_search(request: GlobalSearchRequest):
    """
    Answer questions across all processed videos.
    
    Uses semantic search (embeddings) + keyword search fallback
    to find relevant content, then synthesizes an answer.
    """
    import numpy as np
    from videograph.utils import get_openai_client, resolve_model_name

    try:
        client = get_openai_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    all_results = []
    videos_with_embeddings = 0
    videos_with_keyword_search = 0
    
    # Get query embedding for semantic search
    query_embedding = None
    try:
        response = client.embeddings.create(
            model=resolve_model_name("text-embedding-3-small", "embedding"),
            input=request.query[:8000]
        )
        query_embedding = np.array(response.data[0].embedding)
    except Exception as e:
        logger.warning(f"Failed to compute query embedding: {e}")
    
    # Search across all video directories
    for video_dir in DATA_DIR.iterdir():
        if not video_dir.is_dir():
            continue
        
        metadata_path = video_dir / "metadata.json"
        embeddings_path = video_dir / "embeddings.json"
        graph_path = video_dir / "graph.json"
        
        # Load metadata for video title
        video_title = video_dir.name
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                video_title = metadata.get('metadata', {}).get('title', video_dir.name)
            except:
                pass
        
        # Try semantic search first (if embeddings exist)
        if embeddings_path.exists() and graph_path.exists() and query_embedding is not None:
            videos_with_embeddings += 1
            try:
                with open(embeddings_path, 'r') as f:
                    embeddings = json.load(f)
                
                graph_data = load_json_safe(graph_path)
                nodes_by_id = {n['id']: n for n in graph_data.get('nodes', [])}
                
                for node_id, embedding in embeddings.items():
                    node_emb = np.array(embedding)
                    
                    norm_q = np.linalg.norm(query_embedding)
                    norm_n = np.linalg.norm(node_emb)
                    if norm_q == 0 or norm_n == 0:
                        continue
                    
                    similarity = float(np.dot(query_embedding, node_emb) / (norm_q * norm_n))
                    
                    # Lower threshold for better recall
                    if similarity < 0.25:
                        continue
                    
                    node = nodes_by_id.get(node_id, {})
                    if not node:
                        continue
                    
                    text = node.get('text', '') or node.get('visual_description', '') or node.get('title', '') or ''
                    if not text:
                        continue
                    
                    all_results.append({
                        'video_id': video_dir.name,
                        'video_title': video_title,
                        'text': text[:500],
                        'start': node.get('start', 0) or 0,
                        'score': similarity,
                        'method': 'semantic'
                    })
            except Exception as e:
                logger.warning(f"Failed semantic search on {video_dir.name}: {e}")
        
        # Always do keyword search as well (catches what embeddings might miss)
        videos_with_keyword_search += 1
        keyword_results = keyword_search_video(video_dir, request.query, video_title)
        for r in keyword_results:
            r['method'] = 'keyword'
            # Boost keyword results slightly for exact matches
            r['score'] = r['score'] * 0.8  # Slightly lower than semantic
            all_results.append(r)
    
    # Deduplicate by similar text content
    seen_texts = set()
    unique_results = []
    for r in all_results:
        text_key = r['text'][:100].lower()
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            unique_results.append(r)
    
    # Sort by score and get top results
    unique_results.sort(key=lambda x: x['score'], reverse=True)
    top_results = unique_results[:request.top_k]
    
    total_videos = len([d for d in DATA_DIR.iterdir() if d.is_dir()])
    
    # If no results found
    if not top_results:
        return {
            "query": request.query,
            "answer": f"I couldn't find any relevant information about this topic in the {total_videos} processed videos.",
            "sources": [],
            "videos_searched": total_videos
        }
    
    # Build context for LLM
    context_parts = []
    for r in top_results:
        mins = int(r['start'] // 60)
        secs = int(r['start'] % 60)
        context_parts.append(f"[Video: {r['video_title']}, at {mins}:{secs:02d}] {r['text']}")
    
    context = "\n\n".join(context_parts)
    
    # Generate answer using LLM
    try:
        config = load_config()
        openai_config = config.get("openai", {})
        text_model = resolve_model_name(
            openai_config.get("text_model", "gpt-4o"), "chat"
        )
        temperature = float(openai_config.get("temperature", 0.3))
        
        answer_response = client.chat.completions.create(
            model=text_model,
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant that answers questions based on video content.
                    
Answer the user's question based on the provided video excerpts. Be concise and direct.
At the end of your answer, list the relevant videos in this format:

Sources:
- "Video Title" (at MM:SS)
- "Another Video" (at MM:SS)

Only cite videos that are actually relevant to answering the question."""
                },
                {
                    "role": "user",
                    "content": f"""Question: {request.query}

Video content:
{context}

Provide a concise answer based on this content."""
                }
            ],
            temperature=temperature,
            max_tokens=1024
        )
        answer = answer_response.choices[0].message.content
    except Exception as e:
        logger.error(f"Failed to generate answer: {e}")
        answer = "Failed to generate an answer. Please try again."
    
    # Get unique video sources
    seen_videos = set()
    sources = []
    for r in top_results[:5]:
        if r['video_id'] not in seen_videos:
            seen_videos.add(r['video_id'])
            sources.append({
                "video_id": r['video_id'],
                "video_title": r['video_title'],
                "timestamp": r['start']
            })
    
    log_event("global", "global_qa", {
        "query": request.query,
        "videos_cited": len(sources),
        "semantic_videos": videos_with_embeddings,
        "keyword_videos": videos_with_keyword_search
    })
    
    return {
        "query": request.query,
        "answer": answer,
        "sources": sources,
        "videos_searched": total_videos
    }


# ============ Run Server ============

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import signal
    import sys
    import uvicorn

    # Ensure Ctrl+C works on Windows
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    try:
        uvicorn.run(app, host=host, port=port)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Server shutting down.")


if __name__ == "__main__":
    run_server()
