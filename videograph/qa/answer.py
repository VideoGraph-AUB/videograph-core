"""
Question answering with evidence module.

Returns:
- answer_text
- evidence list [{node_id, clip_id, start, end, snippet}]
- highlighted_subgraph
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from ..graph.models import MultimodalGraph, NodeType
from ..retrieval.graph_retrieval import GraphRetriever, RetrievalResult
from ..cache.openai_cache import get_cache
from ..utils import get_openai_client, resolve_model_name

logger = logging.getLogger(__name__)

NO_CONTEXT_MESSAGE = "No relevant context was retrieved from the video graph."


@dataclass
class Evidence:
    """A piece of evidence supporting an answer."""
    node_id: str
    clip_id: Optional[str]
    start: float
    end: float
    snippet: str
    node_type: str
    relevance_score: float


@dataclass
class QAResponse:
    """Response from the QA system."""
    answer: str
    evidence: List[Evidence]
    subgraph: MultimodalGraph
    confidence: float
    citations: List[str]  # List of [node_id] citations in the answer


class QuestionAnswerer:
    """Answers questions using graph-based retrieval."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        text_model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_evidence: int = 12,
        citation_style: str = "inline",
        include_visual_context: bool = True,
        hybrid_alpha: float = 0.7,
        use_state_change_channel: bool = True,
        expansion_edge_types=None,
        cache_enabled: bool = True
    ):
        """
        Initialize the QA system.
        
        Args:
            api_key: OpenAI API key
            text_model: Model for answer generation
            temperature: Sampling temperature
            max_tokens: Max tokens for answer
            max_evidence: Maximum evidence items to include
            citation_style: "inline" or "footnote"
            include_visual_context: Whether to include visual descriptions
            hybrid_alpha: Weight for semantic similarity versus lexical matching
            use_state_change_channel: Whether to add visual state-change retrieval seeds
            expansion_edge_types: Optional edge-type override for expansion
            cache_enabled: Whether to cache API calls
        """
        self.client = get_openai_client(api_key)
        self.text_model = resolve_model_name(text_model, "chat")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_evidence = max_evidence
        self.citation_style = citation_style
        self.include_visual_context = include_visual_context
        self.use_state_change_channel = use_state_change_channel
        self.expansion_edge_types = expansion_edge_types
        self.cache = get_cache() if cache_enabled else None
        
        self.retriever = GraphRetriever(
            api_key=api_key,
            hybrid_alpha=hybrid_alpha,
            cache_enabled=cache_enabled,
        )
    
    def answer(
        self,
        question: str,
        graph: MultimodalGraph,
        top_k: int = 10,
        hop_expansion: int = 2,
        embeddings_path: Optional[Path] = None
    ) -> QAResponse:
        """
        Answer a question using graph-based retrieval.
        
        Args:
            question: The question to answer
            graph: The multimodal graph
            top_k: Number of nodes to retrieve
            hop_expansion: Hops for subgraph expansion
            embeddings_path: Path to pre-computed embeddings file
            
        Returns:
            QAResponse with answer, evidence, and subgraph
        """
        # Retrieve relevant nodes
        results, subgraph = self.retriever.retrieve(
            question,
            graph,
            top_k=top_k,
            hop_expansion=hop_expansion,
            include_visual=self.include_visual_context,
            embeddings_path=embeddings_path,
            use_state_change_channel=self.use_state_change_channel,
            expansion_edge_types=self.expansion_edge_types,
        )
        
        # Build the same retrieval-grounded context shape used by evaluation:
        # primary retrieved evidence first, then expanded structural evidence.
        video_dir = embeddings_path.parent if embeddings_path else None
        context = build_retrieval_context(
            results,
            subgraph,
            _load_state_change_by_clip(video_dir),
        )

        # Build evidence list from the same retrieval result set returned by the retriever.
        evidence = self._build_evidence(results, subgraph)
        
        # Generate answer
        answer, citations, confidence = self._generate_answer(question, context, evidence)
        
        return QAResponse(
            answer=answer,
            evidence=evidence[:self.max_evidence],
            subgraph=subgraph,
            confidence=confidence,
            citations=citations
        )
    
    def _build_evidence(
        self,
        results: List[RetrievalResult],
        graph: MultimodalGraph
    ) -> List[Evidence]:
        """Build evidence list from retrieval results."""
        evidence = []
        
        for result in results:
            node = graph.nodes.get(result.node_id)
            if not node:
                continue
            
            # Get snippet
            if node.node_type == NodeType.TRANSCRIPT:
                snippet = node.text
            elif node.node_type == NodeType.VISUAL:
                snippet_parts = []
                if node.visual_description:
                    snippet_parts.append(f"[Visual: {node.visual_description}]")
                if node.ocr_text:
                    snippet_parts.append(f"OCR: {node.ocr_text}")
                state_change = getattr(node, "state_change_from_previous", "") or ""
                if state_change:
                    snippet_parts.append(f"State change: {state_change}")
                detected_entities = getattr(node, "detected_entities", []) or []
                if detected_entities:
                    snippet_parts.append(f"Entities: {', '.join(detected_entities[:10])}")
                snippet = " ".join(snippet_parts)
            elif node.node_type == NodeType.TOPIC:
                snippet = f"[Topic: {node.title}] {node.description or ''}"
            elif node.node_type == NodeType.ENTITY:
                aliases = getattr(node, "aliases", []) or []
                snippet = f"[Entity: {node.name}] {getattr(node, 'entity_type', '') or ''}"
                if aliases:
                    snippet += f" Aliases: {', '.join(aliases[:10])}"
            else:
                snippet = ""
            
            # Get clip_id from aligned visual node if this is a transcript
            clip_id = result.clip_id
            if not clip_id and node.node_type == NodeType.TRANSCRIPT:
                # Find aligned visual node
                for edge in graph.edges.values():
                    if edge.source == node.id or edge.target == node.id:
                        other_id = edge.target if edge.source == node.id else edge.source
                        other_node = graph.nodes.get(other_id)
                        if other_node and other_node.node_type == NodeType.VISUAL:
                            clip_id = other_node.clip_id
                            break
            
            # Handle None and infinity values for start/end
            start = result.start if result.start is not None else 0.0
            end = result.end if result.end is not None and result.end != float('inf') else (start + 1.0)
            
            evidence.append(Evidence(
                node_id=result.node_id,
                clip_id=clip_id,
                start=start,
                end=end,
                snippet=snippet[:500],  # Truncate long snippets
                node_type=result.node_type,
                relevance_score=result.score
            ))
        
        return evidence
    
    def _generate_answer(
        self,
        question: str,
        context: str,
        evidence: List[Evidence]
    ) -> Tuple[str, List[str], float]:
        """Generate the answer using LLM."""
        
        system_prompt = f"""You are a video content analyst. Answer questions based on the provided video transcript and visual evidence.

Rules:
1. Only use information from the provided context
2. Cite your sources using [node_id] format, e.g., [t_0001]
3. Include timestamps when relevant
4. If a visual description or transcript directly answers the question, answer directly
5. If the answer is not in the context, say so
6. Be concise but complete

Citation style: {self.citation_style}"""

        user_prompt = f"""Context from video:
{context}

Question: {question}

Provide a detailed answer with citations to the evidence nodes."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # Check cache
        if self.cache:
            cached = self.cache.get(self.text_model, messages, params)
            if cached:
                answer = cached.get("text", "")
            else:
                response = self.client.chat.completions.create(
                    model=self.text_model,
                    messages=messages,
                    **params
                )
                answer = response.choices[0].message.content
                self.cache.set(self.text_model, messages, params, {"text": answer})
        else:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=messages,
                **params
            )
            answer = response.choices[0].message.content
        
        # Extract citations from answer
        import re
        cited = []
        for ev in evidence:
            if f"[{ev.node_id}]" in answer:
                cited.append(ev.node_id)
        if not cited:
            cited = re.findall(r'\[([A-Za-z0-9_:-]+)\]', answer)
        citations = list(dict.fromkeys(cited))
        
        # Calculate confidence based on evidence quality
        # Improved confidence calculation that considers multiple factors
        if evidence:
            # Average relevance score (already normalized to [0, 1])
            avg_score = sum(e.relevance_score for e in evidence[:5]) / min(len(evidence), 5)
            
            # Consider evidence diversity (temporal spread, node type diversity)
            evidence_types = set(e.node_type for e in evidence[:5])
            type_diversity = len(evidence_types) / max(len(evidence), 1)
            
            # Consider citation quality (how many evidence items were cited)
            citation_ratio = len(citations) / min(len(evidence[:5]), len(citations) + 1)
            
            # Weighted confidence: 70% relevance, 20% diversity, 10% citation quality
            confidence = (
                0.7 * avg_score +
                0.2 * type_diversity +
                0.1 * citation_ratio
            )
            
            # Ensure confidence is in [0, 1] range
            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = 0.0
        
        return answer, citations, confidence


def build_retrieval_context(results, subgraph, state_change_by_clip: dict) -> str:
    """Build QA context with primary retrieved evidence and expanded evidence."""
    def sort_key(result) -> tuple:
        return (
            float(getattr(result, "start", 0.0) or 0.0),
            float(getattr(result, "end", 0.0) or 0.0),
            str(getattr(result, "node_type", "")),
            str(getattr(result, "node_id", "")),
        )

    def node_type_value(node) -> str:
        node_type = getattr(node, "node_type", "")
        return node_type.value if hasattr(node_type, "value") else str(node_type)

    def build_snippet(node) -> str:
        node_type = node_type_value(node)

        if node_type == "TranscriptNode":
            return (getattr(node, "text", "") or "")[:1000]

        if node_type == "VisualNode":
            visual_description = getattr(node, "visual_description", "") or ""
            ocr_text = getattr(node, "ocr_text", "") or ""
            detected_entities = getattr(node, "detected_entities", []) or []
            clip_id = getattr(node, "clip_id", "") or ""
            state_change = getattr(node, "state_change_from_previous", "") or ""
            if clip_id and state_change_by_clip:
                state_change = state_change or (state_change_by_clip.get(clip_id, "") or "")

            snippet_parts = []
            if visual_description:
                snippet_parts.append(visual_description)
            if ocr_text:
                snippet_parts.append(f"OCR: {ocr_text}")
            if detected_entities:
                snippet_parts.append(f"Entities: {', '.join(detected_entities[:10])}")
            if state_change.strip():
                snippet_parts.append(f"State change from previous clip: {state_change.strip()}")
            return " ".join(snippet_parts)[:1000]

        if node_type == "TopicNode":
            title = getattr(node, "title", "") or ""
            description = getattr(node, "description", "") or ""
            keywords = getattr(node, "keywords", []) or []
            snippet_parts = []
            if title:
                snippet_parts.append(title)
            if description:
                snippet_parts.append(description)
            if keywords:
                snippet_parts.append(f"Keywords: {', '.join(keywords[:10])}")
            return " ".join(snippet_parts)[:1000]

        if node_type == "EntityNode":
            name = getattr(node, "name", "") or ""
            entity_type = getattr(node, "entity_type", "") or ""
            aliases = getattr(node, "aliases", []) or []
            snippet_parts = []
            if name:
                snippet_parts.append(name)
            if entity_type:
                snippet_parts.append(f"({entity_type})")
            if aliases:
                snippet_parts.append(f"Aliases: {', '.join(aliases[:10])}")
            return " ".join(snippet_parts)[:1000]

        return ""

    def format_entry(result, node) -> str:
        snippet = build_snippet(node)
        if not snippet:
            return ""
        start = getattr(node, "start", 0) or 0
        end = getattr(node, "end", 0) or 0
        ts = f" [{start:.1f}s-{end:.1f}s]" if start or end else ""
        return f"[{result.node_id}] ({node_type_value(node)}{ts}): {snippet}"

    if not subgraph:
        return NO_CONTEXT_MESSAGE

    primary_results = sorted(
        [r for r in results if not getattr(r, "is_expanded", False)],
        key=sort_key,
    )
    expanded_results = sorted(
        [r for r in results if getattr(r, "is_expanded", False)],
        key=sort_key,
    )

    primary_entries = []
    for result in primary_results:
        node = subgraph.nodes.get(result.node_id)
        if node is None:
            continue
        entry = format_entry(result, node)
        if entry:
            primary_entries.append(entry)

    expanded_entries = []
    for result in expanded_results:
        node = subgraph.nodes.get(result.node_id)
        if node is None:
            continue
        entry = format_entry(result, node)
        if not entry:
            continue
        if getattr(result, "expansion_source", None):
            entry = f"{entry}, expanded via {result.expansion_source}"
        expanded_entries.append(entry)

    if not primary_entries and not expanded_entries:
        return NO_CONTEXT_MESSAGE

    sections = []
    if primary_entries:
        sections.append("Primary retrieved evidence:\n" + "\n\n".join(primary_entries))
    if expanded_entries:
        sections.append("Expanded 1-hop evidence:\n" + "\n\n".join(expanded_entries))

    return "\n\n".join(sections)


def _load_state_change_by_clip(video_dir: Optional[Path]) -> dict:
    """Load clip-level state change annotations from sidecar artifacts."""
    if video_dir is None:
        return {}

    state_index_path = Path(video_dir) / "state_changes.json"
    if state_index_path.exists():
        try:
            with open(state_index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            mapping = data.get("state_change_by_clip", {})
            if isinstance(mapping, dict):
                return {
                    str(k).strip(): str(v).strip()
                    for k, v in mapping.items()
                    if str(k).strip() and str(v).strip()
                }
        except Exception:
            pass

    visual_path = Path(video_dir) / "visual.json"
    if not visual_path.exists():
        return {}

    try:
        with open(visual_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    mapping = {}
    rows = data.get("analyses", []) if isinstance(data, dict) else []
    for row in rows:
        clip_id = str(row.get("clip_id", "") or "").strip()
        state_change = str(row.get("state_change_from_previous", "") or "").strip()
        if clip_id and state_change:
            mapping[clip_id] = state_change
    return mapping


def answer_question(
    question: str,
    graph_path: str,
    text_model: str = "gpt-4o",
    temperature: float = 0.0,
    top_k: int = 10,
    hop_expansion: int = 2,
    hybrid_alpha: float = 0.7,
    use_state_change_channel: bool = True,
    expansion_edge_types=None,
) -> dict:
    """
    Convenience function for question answering.
    
    Args:
        question: The question to answer
        graph_path: Path to graph.json
        text_model: Model for answer generation
        temperature: Sampling temperature
        top_k: Number of nodes to retrieve
        hop_expansion: Hops for expansion
        hybrid_alpha: Weight for semantic similarity versus lexical matching
        use_state_change_channel: Whether to add visual state-change retrieval seeds
        expansion_edge_types: Optional edge-type override for expansion
        
    Returns:
        Dictionary with answer, evidence, and subgraph
    """
    from ..graph.serialization import load_graph_json
    from pathlib import Path
    
    try:
        graph = load_graph_json(graph_path)
    except Exception as e:
        logger.error(f"Failed to load graph from {graph_path}: {e}")
        raise
    
    qa = QuestionAnswerer(
        text_model=text_model,
        temperature=temperature,
        hybrid_alpha=hybrid_alpha,
        use_state_change_channel=use_state_change_channel,
        expansion_edge_types=expansion_edge_types,
    )
    
    # Pass embeddings_path to retriever if it exists
    graph_path_obj = Path(graph_path)
    embeddings_path = graph_path_obj.parent / "embeddings.json"
    if not embeddings_path.exists():
        embeddings_path = None
        logger.info(f"Embeddings file not found at {embeddings_path}, will compute on-demand")
    else:
        logger.info(f"Using pre-computed embeddings from {embeddings_path}")
    
    try:
        response = qa.answer(question, graph, top_k=top_k, hop_expansion=hop_expansion, embeddings_path=embeddings_path)
    except Exception as e:
        logger.error(f"Error during question answering: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return {
        "answer": response.answer,
        "confidence": response.confidence,
        "citations": response.citations,
        "evidence": [asdict(e) for e in response.evidence],
        "subgraph": response.subgraph.to_dict()
    }


