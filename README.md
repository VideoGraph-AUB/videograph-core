# VideoGraph Core

Core backend and graph-based video QA engine for VideoGraph.

This package handles the full pipeline from raw video to queryable knowledge graph:

- Adaptive video ingestion with scene detection and keyframe extraction
- Audio transcription via OpenAI Whisper
- Visual captioning and OCR via OpenAI vision models
- Multimodal knowledge graph construction (transcript, visual, entity, and topic nodes)
- Hybrid graph retrieval (semantic + lexical) with hop expansion
- Question answering over the constructed graph
- FastAPI server used by the VideoGraph UI

## Prerequisites

- **Python >= 3.10**
- **FFmpeg** and **FFprobe** on `PATH` - used for audio extraction, video clipping, and compression
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to `PATH`
- **OpenAI API key** - required for transcription (Whisper), vision captioning (GPT-4o), embeddings, and QA

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env         # then set OPENAI_API_KEY
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env       # then set OPENAI_API_KEY
```

## CLI

```bash
python -m videograph --help
```

Main commands:

| Command | Description |
|---------|-------------|
| `build` | Build a knowledge graph from a YouTube video |
| `serve` | Start the FastAPI backend server |
| `query` | Query an existing video graph |

## Run the API Server

```bash
python -m videograph serve --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/api/health
```

## Configuration

All pipeline parameters (models, scene detection thresholds, retrieval settings,
parallelism) are configured in [`config/default.yaml`](config/default.yaml).

## Docker

The Docker image installs FFmpeg automatically:

```bash
docker build -t videograph-core .
docker run --env-file .env -p 8000:8000 -v ./data:/app/data videograph-core
```

## Repository Boundary

This repo is the backend/core package only. Benchmark evaluation code lives in
`videograph-evaluation`, the UI lives in `videograph-frontend`, and the
one-command deployment launcher lives in `videograph`.
