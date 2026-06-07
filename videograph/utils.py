import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

from openai import OpenAI


_OCR_FAILURE_PATTERNS = (
    "no text found",
    "no readable text found",
    "sorry, i can't",
    "sorry, i cant",
    "can't extract text",
    "cannot extract text",
)
_NO_CHANGE_PATTERNS = {
    "no change",
    "no significant change",
    "same scene",
    "none",
    "n/a",
    "na",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def hash_payload(payload: Dict[str, Any]) -> str:
    if HAS_ORJSON:
        serialized = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    else:
        serialized = json.dumps(payload, sort_keys=True).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()


def cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def load_cache(cache_dir: Path, key: str) -> Optional[Dict[str, Any]]:
    path = cache_path(cache_dir, key)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(cache_dir: Path, key: str, data: Dict[str, Any]) -> None:
    ensure_dir(cache_dir)
    path = cache_path(cache_dir, key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start

    return wrapper


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    return OpenAI(api_key=api_key)


def log_jsonl(path: Path, record: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        if HAS_ORJSON:
            f.write(orjson.dumps(record).decode("utf-8") + "\n")
        else:
            f.write(json.dumps(record) + "\n")


def normalize_text(text: Any) -> str:
    """Collapse whitespace while preserving the semantic content of a short text."""
    return " ".join(str(text or "").split()).strip()


def sanitize_state_change_text(text: Any) -> str:
    """Drop boilerplate state-change outputs that do not add retrieval value."""
    cleaned = normalize_text(text)
    if cleaned.lower() in _NO_CHANGE_PATTERNS:
        return ""
    return cleaned


def sanitize_ocr_text(text: Any) -> str:
    """Remove OCR boilerplate/refusal text while keeping useful visible text."""
    if not text:
        return ""

    seen = set()
    cleaned_lines = []
    for raw_line in str(text).splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue
        low = line.lower()
        if any(pattern in low for pattern in _OCR_FAILURE_PATTERNS):
            continue
        if line in seen:
            continue
        seen.add(line)
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def sanitize_entity_strings(items: Any) -> list[str]:
    """Normalize and deduplicate short entity strings."""
    if not isinstance(items, list):
        return []

    seen = set()
    cleaned = []
    for item in items:
        text = normalize_text(item)
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def get_visual_description_text_for_embedding(node) -> str:
    """Text used for the primary visual-content embedding of a visual node."""
    return str(getattr(node, "visual_description", "") or "")


def get_visual_state_change_text_for_embedding(node) -> str:
    """Text used for the temporal-change embedding of a visual node."""
    state_change = str(getattr(node, "state_change_from_previous", "") or "")
    if not state_change:
        return ""
    return f"Temporal change from previous clip: {state_change}"


def get_node_text_for_embedding(node) -> str:
    """
    Standardized function to extract text representation of a node for embedding computation.
    
    This ensures consistency between:
    - Pre-computed embeddings (during graph building)
    - On-demand embeddings (during retrieval)
    - Lexical scoring (during retrieval)
    
    Args:
        node: A node object (TranscriptNode, VisualNode, TopicNode, or EntityNode)
        
    Returns:
        Text string representation of the node
    """
    from .graph.models import NodeType
    
    if node.node_type == NodeType.TRANSCRIPT:
        return node.text
    
    elif node.node_type == NodeType.VISUAL:
        parts = []

        visual_description = str(node.visual_description or "")
        if visual_description:
            parts.append(visual_description)

        state_change = str(getattr(node, "state_change_from_previous", "") or "")
        if state_change:
            parts.append(f"Temporal change from previous clip: {state_change}")

        ocr_text = str(node.ocr_text or "")
        if ocr_text:
            parts.append(f"Visible text: {ocr_text.replace(chr(10), '; ')}")

        detected_entities = node.detected_entities or []
        if detected_entities:
            parts.append(f"Key entities: {', '.join(detected_entities[:15])}")

        return " ".join(parts)
    
    elif node.node_type == NodeType.TOPIC:
        parts = [node.title]
        if node.description:
            parts.append(node.description)
        if node.keywords:
            parts.append(", ".join(node.keywords))
        return ". ".join(parts)
    
    elif node.node_type == NodeType.ENTITY:
        parts = [f"{node.name} ({node.entity_type})"]
        if node.aliases:
            parts.append(f"Aliases: {', '.join(node.aliases)}")
        return ". ".join(parts)
    
    else:
        return ""


