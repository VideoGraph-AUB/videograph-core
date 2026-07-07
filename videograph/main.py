"""
Main entry point for VideoGraph.

Usage:
    python -m videograph build --youtube_url <URL> [--output_dir <dir>]
    python -m videograph serve [--host <host>] [--port <port>]
    python -m videograph query --video_id <id> --query <query>
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from config/default.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def cmd_build(args):
    """Build a graph from a YouTube video."""
    from .video.io import VideoDownloader, extract_video_id
    from .video.adaptive_ingest import process_local_video_adaptive
    from .video.transcribe import transcribe_audio
    from .visual.adaptive_processing import (
        analyze_adaptive_clips,
        update_adaptive_visual_json_with_ocr,
    )
    from .graph.builder import build_video_graph
    
    config = load_config()
    
    # Extract video ID
    video_id = extract_video_id(args.youtube_url)
    output_dir = Path(args.output_dir) / video_id if args.output_dir else Path("data/videos") / video_id
    
    logger.info(f"Processing video: {video_id}")
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Download, adaptive scene clipping, and adaptive keyframes
    logger.info("Step 1/5: Downloading video and running adaptive preprocessing...")
    downloader = VideoDownloader(output_base=str(output_dir.parent))
    video_path, download_metadata = downloader.download_video(args.youtube_url)
    process_local_video_adaptive(
        video_path=str(video_path),
        output_dir=str(output_dir),
        video_id=video_id,
        config=config,
    )
    metadata_path = output_dir / "metadata.json"
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
    
    # Step 2: Transcribe
    logger.info("Step 2/5: Transcribing audio...")
    audio_path = output_dir / "audio.wav"
    trans_config = config.get("transcription", {})
    transcribe_audio(
        str(audio_path),
        output_dir=str(output_dir),
        model=config.get("openai", {}).get("transcription_model", "whisper-1"),
        language=trans_config.get("language"),
        timestamp_granularity=trans_config.get("timestamp_granularity", "segment"),
        filter_hallucinations=trans_config.get("filter_hallucinations", True),
        no_speech_threshold=trans_config.get("no_speech_threshold", 0.6),
        logprob_threshold=trans_config.get("logprob_threshold", -1.0),
        compression_ratio_threshold=trans_config.get("compression_ratio_threshold", 2.4)
    )
    
    # Step 3: Adaptive visual captioning
    logger.info("Step 3/5: Analyzing adaptive clip keyframes...")
    visual_config = config.get("visual", {})
    analyze_adaptive_clips(
        str(output_dir),
        model=config.get("openai", {}).get("vision_model", "gpt-4o"),
        prompt_style=visual_config.get("vision_prompt_style", "detailed"),
        temperature=float(config.get("openai", {}).get("temperature", 0.0)),
        max_parallel=int(config.get("processing", {}).get("max_parallel_vision", 5)),
        append_state_change_to_description=bool(
            visual_config.get("append_state_change_to_description", False)
        ),
    )
    
    # Step 4: OCR (via OpenAI Vision)
    logger.info("Step 4/5: Extracting on-screen text (via Vision API)...")
    if visual_config.get("ocr_enabled", True):
        try:
            update_adaptive_visual_json_with_ocr(
                str(output_dir),
                model=config.get("openai", {}).get("vision_model", "gpt-4o"),
                max_parallel=int(config.get("processing", {}).get("max_parallel_vision", 5)),
            )
        except Exception as e:
            logger.warning(f"OCR failed (non-fatal): {e}")
    
    # Step 4b: GRL — graph reinforcement (critique -> targeted re-perception -> gated
    # write-back). Enriches clip captions before summary synthesis; rebuild deferred.
    grl_cfg = config.get("graph", {}).get("reinforcement", {})
    if grl_cfg.get("enabled", False):
        from .graph.reinforce import reinforce_video_graph
        try:
            reinforce_video_graph(
                str(output_dir),
                text_model=config.get("openai", {}).get("text_model", "gpt-4o"),
                vision_model=config.get("openai", {}).get("vision_model", "gpt-4o"),
                max_probes=int(grl_cfg.get("max_probes", 5)),
                frames_per_probe=int(grl_cfg.get("frames_per_probe", 8)),
                rebuild=False,
            )
        except Exception as e:
            logger.warning(f"GRL failed (non-fatal): {e}")

    # Step 4c: Multi-granularity — whole-video summary node (coarse level alongside
    # event-granular clips; retrieval routes by similarity)
    from .visual.adaptive_processing import append_video_summary_node
    try:
        append_video_summary_node(
            str(output_dir),
            model=config.get("openai", {}).get("text_model", "gpt-4o"),
        )
    except Exception as e:
        logger.warning(f"Summary node failed (non-fatal): {e}")

    # Step 5: Build graph
    logger.info("Step 5/5: Building knowledge graph...")
    graph_data = build_video_graph(str(output_dir), config=config)
    
    # Summary
    stats = graph_data.get("stats", {})
    logger.info("=" * 50)
    logger.info("Processing complete!")
    logger.info(f"Total nodes: {stats.get('total_nodes', 0)}")
    logger.info(f"Total edges: {stats.get('total_edges', 0)}")
    logger.info(f"Nodes by type: {stats.get('nodes_by_type', {})}")
    logger.info(f"Edges by type: {stats.get('edges_by_type', {})}")
    logger.info(f"Graph saved to: {output_dir / 'graph.json'}")
    
    return output_dir


def cmd_serve(args):
    """Start the API server."""
    from .api.server import run_server
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    run_server(host=args.host, port=args.port)


def cmd_query(args):
    """Query a video graph."""
    from .qa.answer import answer_question
    config = load_config()
    
    graph_path = Path("data/videos") / args.video_id / "graph.json"
    
    if not graph_path.exists():
        logger.error(f"Graph not found: {graph_path}")
        sys.exit(1)
    
    logger.info(f"Querying graph: {graph_path}")
    logger.info(f"Query: {args.query}")
    
    retrieval_config = config.get("retrieval", {})
    result = answer_question(
        args.query,
        str(graph_path),
        text_model=config.get("openai", {}).get("text_model", "gpt-4o"),
        temperature=float(config.get("openai", {}).get("temperature", 0.0)),
        top_k=args.top_k if args.top_k is not None else int(retrieval_config.get("top_k", 10)),
        hop_expansion=(
            args.hop_expansion
            if args.hop_expansion is not None
            else int(retrieval_config.get("hop_expansion", 2))
        ),
        hybrid_alpha=float(retrieval_config.get("hybrid_alpha", 0.7)),
    )
    
    print("\n" + "=" * 50)
    print("ANSWER:")
    print("=" * 50)
    print(result["answer"])
    
    print("\n" + "=" * 50)
    print("EVIDENCE:")
    print("=" * 50)
    for i, ev in enumerate(result["evidence"], 1):
        print(f"\n[{i}] {ev['node_id']} ({ev['start']:.1f}s - {ev['end']:.1f}s)")
        print(f"    {ev['snippet'][:200]}...")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Full result saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="VideoGraph - multimodal video understanding"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build graph from YouTube video")
    build_parser.add_argument("--youtube_url", "-u", required=True, help="YouTube video URL")
    build_parser.add_argument("--output_dir", "-o", help="Output directory (default: data/videos/)")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    serve_parser.add_argument("--port", "-p", type=int, default=8000, help="Port (default: 8000)")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query a video graph")
    query_parser.add_argument("--video_id", "-v", required=True, help="Video ID")
    query_parser.add_argument("--query", "-q", required=True, help="Query string")
    query_parser.add_argument("--top_k", "-k", type=int, default=None, help="Top-k results")
    query_parser.add_argument("--hop_expansion", type=int, default=None, help="Hop expansion")
    query_parser.add_argument("--output", "-o", help="Output file for full result JSON")
    
    args = parser.parse_args()
    
    if args.command == "build":
        cmd_build(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()



