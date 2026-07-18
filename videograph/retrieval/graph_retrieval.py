"""
Graph-based retrieval module.

Given a query:
1. Compute relevance over nodes (embedding + lexical)
2. Return top-k TranscriptNodes with k-hop expansion
3. Include aligned VisualNodes
"""

import logging
import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Set, Tuple
import numpy as np

from ..graph.models import (
    MultimodalGraph, BaseNode,
    NodeType, EdgeType
)
from ..cache.openai_cache import get_cache
from ..utils import (
    get_node_text_for_embedding,
    get_openai_client,
    get_visual_description_text_for_embedding,
    get_visual_state_change_text_for_embedding,
    resolve_model_name,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of a retrieval query."""
    node_id: str
    node_type: str
    text: str
    start: float
    end: float
    score: float
    clip_id: Optional[str] = None
    is_expanded: bool = False
    expansion_source: Optional[str] = None


class GraphRetriever:
    """Retrieves relevant subgraphs for queries."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 10,
        hop_expansion: int = 2,
        hybrid_alpha: float = 0.7,
        cache_enabled: bool = True,
        persist_visual_channel_embeddings: bool = True,
    ):
        """
        Initialize the retriever.
        
        Args:
            api_key: OpenAI API key
            embedding_model: Model for computing embeddings
            top_k: Number of top nodes to retrieve
            hop_expansion: Number of hops for subgraph expansion
            hybrid_alpha: Weight for embedding similarity (1-alpha for lexical)
            cache_enabled: Whether to cache embeddings
            persist_visual_channel_embeddings: Whether to write missing visual sidecar embeddings
        """
        self.client = get_openai_client(api_key)
        self.embedding_model = resolve_model_name(embedding_model, "embedding")
        self.top_k = top_k
        self.hop_expansion = hop_expansion
        self.hybrid_alpha = hybrid_alpha
        self.cache = get_cache() if cache_enabled else None
        self.persist_visual_channel_embeddings = persist_visual_channel_embeddings
        
        # Precomputed embeddings cache
        self._node_embeddings: Dict[str, np.ndarray] = {}
        self._visual_description_embeddings: Dict[str, np.ndarray] = {}
        self._visual_state_change_embeddings: Dict[str, np.ndarray] = {}
        self._visual_state_change_text_overrides: Dict[str, str] = {}
        self._node_embeddings_source: Optional[str] = None
        self._visual_channel_embeddings_source: Optional[str] = None
        self._visual_state_change_overrides_source: Optional[str] = None

    def _normalize_node_types(
        self,
        node_types: Optional[Iterable],
    ) -> Optional[Set[NodeType]]:
        """Normalize node-type filters supplied by ablation runners."""
        if node_types is None:
            return None

        normalized: Set[NodeType] = set()
        for node_type in node_types:
            if isinstance(node_type, NodeType):
                normalized.add(node_type)
                continue

            raw = str(node_type).strip()
            if not raw:
                continue

            try:
                normalized.add(NodeType(raw))
                continue
            except ValueError:
                pass

            for candidate in NodeType:
                if raw.lower() in {candidate.name.lower(), candidate.value.lower()}:
                    normalized.add(candidate)
                    break

        return normalized
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string."""
        if not text.strip():
            return np.zeros(1536)  # Default dimension for text-embedding-3-small
        
        # Check cache
        cache_key = {"text": text[:1000], "model": self.embedding_model}
        if self.cache:
            cached = self.cache.get(
                self.embedding_model,
                [{"role": "embed", "content": text[:100]}],
                cache_key
            )
            if cached:
                emb = np.array(cached.get("embedding", []))
                # Validate cached dimensionality: a historic truncated write (e.g. 1534
                # of 1536 dims) would otherwise be replayed forever and crash retrieval.
                expected = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}.get(self.embedding_model)
                if expected is None or emb.shape == (expected,):
                    return emb
                logger.warning(f"Discarding corrupt cached embedding ({emb.shape[0]} dims, expected {expected})")
        
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text[:8000]  # Limit input length
        )
        embedding = response.data[0].embedding
        
        # Cache the result
        if self.cache:
            self.cache.set(
                self.embedding_model,
                [{"role": "embed", "content": text[:100]}],
                cache_key,
                {"embedding": embedding}
            )
        
        return np.array(embedding)
    
    def _load_visual_state_change_overrides(
        self,
        embeddings_path: Optional[Path],
    ) -> None:
        """Load clip-level state-change text from visual.json for older graphs."""
        if not embeddings_path:
            if self._visual_state_change_overrides_source is not None:
                self._visual_state_change_text_overrides = {}
                self._visual_state_change_overrides_source = None
            return

        visual_path = embeddings_path.with_name("visual.json")
        source = str(visual_path.resolve())
        if self._visual_state_change_overrides_source == source:
            return

        self._visual_state_change_text_overrides = {}
        self._visual_state_change_overrides_source = source

        if not visual_path.exists():
            return

        try:
            with open(visual_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load visual state changes from {visual_path}: {e}")
            return

        if isinstance(rows, dict):
            rows = rows.get("analyses", [])
        if not isinstance(rows, list):
            return

        for row in rows:
            if not isinstance(row, dict):
                continue
            clip_id = str(row.get("clip_id", "") or "").strip()
            if not clip_id:
                continue
            state_change = str(row.get("state_change_from_previous", "") or "").strip()
            if not state_change:
                continue
            self._visual_state_change_text_overrides[f"v_{clip_id}"] = state_change

    def _get_visual_state_change_text(self, node_id: str, node: BaseNode) -> str:
        """Return the best available state-change text for a visual node."""
        state_text = get_visual_state_change_text_for_embedding(node)
        if state_text:
            return state_text

        fallback = self._visual_state_change_text_overrides.get(node_id, "")
        if not fallback:
            return ""
        return f"Temporal change from previous clip: {fallback}"

    def _save_visual_channel_embeddings(
        self,
        graph: MultimodalGraph,
        output_path: Path,
    ) -> None:
        """Persist the currently known visual channel embeddings."""
        visual_embeddings_payload: Dict[str, dict] = {}

        for node_id, node in graph.nodes.items():
            if node.node_type != NodeType.VISUAL:
                continue

            channel_payload = {}
            if node_id in self._visual_description_embeddings:
                channel_payload["visual_description"] = (
                    self._visual_description_embeddings[node_id].tolist()
                )
            if node_id in self._visual_state_change_embeddings:
                channel_payload["state_change"] = (
                    self._visual_state_change_embeddings[node_id].tolist()
                )
            if channel_payload:
                visual_embeddings_payload[node_id] = channel_payload

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(visual_embeddings_payload, f)
        except Exception as e:
            logger.warning(f"Failed to save visual channel embeddings to {output_path}: {e}")

    def _compute_visual_channel_embeddings(
        self,
        graph: MultimodalGraph,
        output_path: Optional[Path] = None,
    ) -> None:
        """Compute or refresh sidecar embeddings for visual description/state-change channels."""
        for node_id, node in graph.nodes.items():
            if node.node_type != NodeType.VISUAL:
                continue

            visual_text = get_visual_description_text_for_embedding(node)
            if visual_text.strip():
                emb = self._get_embedding(visual_text)
                self._visual_description_embeddings[node_id] = emb

            state_text = self._get_visual_state_change_text(node_id, node)
            if state_text.strip():
                emb = self._get_embedding(state_text)
                self._visual_state_change_embeddings[node_id] = emb

        if output_path:
            self._save_visual_channel_embeddings(graph, output_path)

    def _load_visual_channel_embeddings(
        self,
        graph: MultimodalGraph,
        embeddings_path: Optional[Path],
    ) -> None:
        """Load sidecar visual embeddings or compute them on demand when missing."""
        visual_embeddings_path: Optional[Path] = None
        if embeddings_path:
            visual_embeddings_path = embeddings_path.with_name("visual_channel_embeddings.json")
        self._load_visual_state_change_overrides(embeddings_path)

        source = (
            str(visual_embeddings_path.resolve())
            if visual_embeddings_path
            else f"graph:{id(graph)}:visual_channels"
        )
        if self._visual_channel_embeddings_source == source:
            return

        self._visual_description_embeddings = {}
        self._visual_state_change_embeddings = {}
        self._visual_channel_embeddings_source = source

        loaded = False
        if visual_embeddings_path and visual_embeddings_path.exists():
            try:
                with open(visual_embeddings_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                for node_id, channels in payload.items():
                    if not isinstance(channels, dict):
                        continue
                    if "visual_description" in channels:
                        self._visual_description_embeddings[node_id] = np.array(channels["visual_description"])
                    if "state_change" in channels:
                        self._visual_state_change_embeddings[node_id] = np.array(channels["state_change"])
                loaded = True
                logger.info(
                    "Loaded visual channel embeddings: "
                    f"{len(self._visual_description_embeddings)} descriptions, "
                    f"{len(self._visual_state_change_embeddings)} state changes"
                )
            except Exception as e:
                logger.warning(f"Failed to load visual channel embeddings: {e}")

        missing_state_change_nodes = [
            node_id
            for node_id, node in graph.nodes.items()
            if node.node_type == NodeType.VISUAL
            and node_id not in self._visual_state_change_embeddings
            and self._get_visual_state_change_text(node_id, node)
        ]

        if not loaded:
            logger.info("Computing visual channel embeddings on demand...")
            output_path = (
                visual_embeddings_path
                if self.persist_visual_channel_embeddings
                else None
            )
            self._compute_visual_channel_embeddings(graph, output_path)
            logger.info(
                "Computed visual channel embeddings: "
                f"{len(self._visual_description_embeddings)} descriptions, "
                f"{len(self._visual_state_change_embeddings)} state changes"
            )
        elif missing_state_change_nodes:
            logger.info(
                f"Refreshing {len(missing_state_change_nodes)} missing state-change embeddings "
                "from visual.json..."
            )
            for node_id in missing_state_change_nodes:
                node = graph.nodes[node_id]
                state_text = self._get_visual_state_change_text(node_id, node)
                if not state_text:
                    continue
                self._visual_state_change_embeddings[node_id] = self._get_embedding(state_text)
            if visual_embeddings_path and self.persist_visual_channel_embeddings:
                self._save_visual_channel_embeddings(graph, visual_embeddings_path)
            logger.info(
                "Updated visual channel embeddings: "
                f"{len(self._visual_description_embeddings)} descriptions, "
                f"{len(self._visual_state_change_embeddings)} state changes"
            )

    def prepare_visual_channel_embeddings(
        self,
        graph: MultimodalGraph,
        embeddings_path: Path,
    ) -> None:
        """Ensure reusable visual-description/state-change embeddings exist."""
        self._load_visual_channel_embeddings(graph, embeddings_path)
    
    def _compute_node_embeddings(self, graph: MultimodalGraph, embeddings_path: Path = None):
        """
        Load or compute embeddings for all nodes.
        
        If embeddings_path is provided and exists, loads pre-computed embeddings.
        Otherwise, computes them on-demand.
        """
        source = (
            str(embeddings_path.resolve())
            if embeddings_path
            else f"graph:{id(graph)}"
        )
        if self._node_embeddings_source != source:
            self._node_embeddings = {}
            self._node_embeddings_source = source

        # Try to load pre-computed embeddings
        if embeddings_path and embeddings_path.exists() and not self._node_embeddings:
            try:
                with open(embeddings_path, 'r') as f:
                    stored_embeddings = json.load(f)
                for node_id, embedding in stored_embeddings.items():
                    self._node_embeddings[node_id] = np.array(embedding)
                logger.info(f"Loaded {len(self._node_embeddings)} pre-computed embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")
        
        if not self._node_embeddings:
            logger.info("Computing node embeddings...")
            
            for node_id, node in graph.nodes.items():
                if node_id in self._node_embeddings:
                    continue
                
                # Get text representation using standardized function
                text = get_node_text_for_embedding(node)
                
                if text:
                    self._node_embeddings[node_id] = self._get_embedding(text)
            
            logger.info(f"Computed {len(self._node_embeddings)} embeddings")

        self._load_visual_channel_embeddings(graph, embeddings_path)
    
    def _lexical_score(self, query: str, text: str) -> float:
        """Compute lexical similarity using term overlap."""
        if not text:
            return 0.0
        
        # Simple word overlap
        query_words = set(re.findall(r'\w+', query.lower()))
        text_words = set(re.findall(r'\w+', text.lower()))
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & text_words)
        return overlap / len(query_words)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if a.size == 0 or b.size == 0:
            return 0.0
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _best_expansion_source(
        self,
        node_id: str,
        seed_ids: set,
        graph: MultimodalGraph,
    ) -> Optional[str]:
        """Describe the approved structural reason an expanded node was included."""
        edge_priority = {
            EdgeType.ALIGNED_TO: 0,
            EdgeType.TEMPORAL_NEXT: 1,
        }
        labels = {
            EdgeType.ALIGNED_TO: "aligned",
            EdgeType.TEMPORAL_NEXT: "temporal_neighbor",
        }

        best: Optional[Tuple[int, str]] = None
        for edge in graph.edges.values():
            if edge.source == node_id:
                other = edge.target
            elif edge.target == node_id:
                other = edge.source
            else:
                continue

            if other not in seed_ids:
                continue

            priority = edge_priority.get(edge.edge_type, 99)
            label = labels.get(edge.edge_type, str(edge.edge_type))
            candidate = (priority, label)
            if best is None or candidate < best:
                best = candidate

        return best[1] if best else None

    def _expanded_sort_key(
        self,
        node: BaseNode,
        score: float,
        expansion_source: Optional[str],
    ) -> Tuple[int, float, int, float]:
        """Prefer aligned evidence first, then keep a stable chronological order."""
        source_priority = {
            "aligned": 0,
            "temporal_neighbor": 1,
            None: 2,
        }
        node_priority = {
            NodeType.VISUAL: 0,
            NodeType.TRANSCRIPT: 1,
            NodeType.TOPIC: 2,
            NodeType.ENTITY: 3,
        }
        return (
            source_priority.get(expansion_source, 2),
            node.start if node.start is not None else float("inf"),
            node_priority.get(node.node_type, 9),
            -score,
        )
    
    def retrieve(
        self,
        query: str,
        graph: MultimodalGraph,
        top_k: Optional[int] = None,
        hop_expansion: Optional[int] = None,
        include_visual: bool = True,
        embeddings_path: Optional[Path] = None,
        allowed_node_types: Optional[Iterable] = None,
        use_state_change_channel: bool = True,
        expansion_edge_types: Optional[Iterable] = None,
    ) -> Tuple[List[RetrievalResult], MultimodalGraph]:
        """
        Retrieve relevant nodes for a query.
        
        Args:
            query: The search query
            graph: The multimodal graph to search
            top_k: Override for number of top results
            hop_expansion: Override for hop expansion
            include_visual: Whether to include aligned visual nodes
            embeddings_path: Path to pre-computed embeddings file
            allowed_node_types: Optional node-type filter for retrieval/context
            use_state_change_channel: Whether to add extra visual state-change seeds
            expansion_edge_types: Optional override for graph expansion edge types
            
        Returns:
            Tuple of (retrieval_results, highlighted_subgraph)
        """
        top_k = self.top_k if top_k is None else top_k
        hop_expansion = self.hop_expansion if hop_expansion is None else hop_expansion
        allowed_types = self._normalize_node_types(allowed_node_types)
        
        # Load or compute embeddings
        self._compute_node_embeddings(graph, embeddings_path)
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Score all nodes
        scores = []
        state_change_scores = []
        
        for node_id, node in graph.nodes.items():
            if allowed_types is not None and node.node_type not in allowed_types:
                continue

            # Get node text using standardized function (ensures consistency with embeddings)
            text = get_node_text_for_embedding(node)
            
            # Compute hybrid score
            embedding_score = 0.0
            if node.node_type == NodeType.VISUAL and node_id in self._visual_description_embeddings:
                embedding_score = self._cosine_similarity(
                    query_embedding,
                    self._visual_description_embeddings[node_id]
                )
                embedding_score = max(0.0, min(1.0, embedding_score))
            elif node_id in self._node_embeddings:
                embedding_score = self._cosine_similarity(
                    query_embedding,
                    self._node_embeddings[node_id]
                )
                # Clamp embedding score to [0, 1] to handle edge cases
                embedding_score = max(0.0, min(1.0, embedding_score))
            
            lexical_score = self._lexical_score(query, text)
            
            # Weighted combination (both scores are now in [0, 1])
            score = self.hybrid_alpha * embedding_score + (1 - self.hybrid_alpha) * lexical_score
            
            # Boost transcript nodes slightly (transcript-first approach)
            # Normalize after boost to keep score in [0, 1] range
            if node.node_type == NodeType.TRANSCRIPT:
                score = min(1.0, score * 1.1)
            
            scores.append((node_id, score, node))
        
            if node.node_type == NodeType.VISUAL and node_id in self._visual_state_change_embeddings:
                state_change_text = self._get_visual_state_change_text(node_id, node)
                state_embedding_score = self._cosine_similarity(
                    query_embedding,
                    self._visual_state_change_embeddings[node_id]
                )
                state_embedding_score = max(0.0, min(1.0, state_embedding_score))
                state_lexical_score = self._lexical_score(query, state_change_text)
                state_score = (
                    self.hybrid_alpha * state_embedding_score
                    + (1 - self.hybrid_alpha) * state_lexical_score
                )
                state_change_scores.append((node_id, state_score, node))
        
        # Sort by score and keep the pass2-style broad seed set.
        scores.sort(key=lambda x: x[1], reverse=True)
        top_nodes = scores[:top_k]
        score_lookup = {node_id: score for node_id, score, _ in scores}
        seed_ids = set(node_id for node_id, _, _ in top_nodes)

        # Add extra visual seeds from the state-change channel.
        if use_state_change_channel:
            state_change_scores.sort(key=lambda x: x[1], reverse=True)
            state_change_seed_budget = max(1, top_k // 2)
            extra_state_change_nodes = []
            for node_id, state_score, node in state_change_scores:
                if node_id in seed_ids:
                    continue
                extra_state_change_nodes.append((node_id, state_score, node))
                if len(extra_state_change_nodes) >= state_change_seed_budget:
                    break
            for node_id, state_score, _ in extra_state_change_nodes:
                seed_ids.add(node_id)
                score_lookup[node_id] = max(score_lookup.get(node_id, 0.0), state_score)
            top_nodes.extend(extra_state_change_nodes)

        # Expand with k-hop neighbors
        expanded_ids = set(seed_ids)
        active_expansion_edge_types = []
        
        if hop_expansion > 0:
            if expansion_edge_types is None:
                active_expansion_edge_types = [EdgeType.TEMPORAL_NEXT]
                if include_visual:
                    active_expansion_edge_types.append(EdgeType.ALIGNED_TO)
            else:
                active_expansion_edge_types = [
                    edge_type if isinstance(edge_type, EdgeType) else EdgeType(str(edge_type))
                    for edge_type in expansion_edge_types
                ]
            
            frontier_ids = set(expanded_ids)
            for _ in range(hop_expansion):
                if not frontier_ids:
                    break

                # Expand one hop from the current frontier, then keep only unseen nodes.
                new_ids = set(
                    graph.expand_subgraph(
                        list(frontier_ids),
                        hops=1,
                        edge_types=active_expansion_edge_types,
                    )
                )
                if allowed_types is not None:
                    new_ids = {
                        node_id
                        for node_id in new_ids
                        if node_id in graph.nodes
                        and graph.nodes[node_id].node_type in allowed_types
                    }
                next_frontier = new_ids - expanded_ids
                if not next_frontier:
                    break

                expanded_ids.update(next_frontier)
                frontier_ids = next_frontier

        if allowed_types is not None:
            expanded_ids = {
                node_id
                for node_id in expanded_ids
                if node_id in graph.nodes
                and graph.nodes[node_id].node_type in allowed_types
            }
        
        # Build seed retrieval results
        results = []
        for node_id, score, node in top_nodes:
            # Use standardized function to get text representation
            text = get_node_text_for_embedding(node)
            
            result = RetrievalResult(
                node_id=node_id,
                node_type=node.node_type.value,
                text=text,
                start=node.start if node.start is not None else 0.0,
                end=node.end if node.end is not None and node.end != float('inf') else 0.0,
                score=score,
                clip_id=node.clip_id if hasattr(node, 'clip_id') else None,
                is_expanded=False,
                expansion_source=None,
            )
            results.append(result)

        # Add expanded nodes after the original seed results so the QA prompt
        # can consume both the direct hits and their 1-hop neighborhood.
        expanded_results = []
        for node_id in expanded_ids - seed_ids:
            node = graph.nodes.get(node_id)
            if node is None:
                continue

            text = get_node_text_for_embedding(node)
            expansion_source = self._best_expansion_source(node_id, seed_ids, graph)
            expanded_results.append(
                (
                    self._expanded_sort_key(
                        node,
                        score_lookup.get(node_id, 0.0),
                        expansion_source,
                    ),
                    RetrievalResult(
                        node_id=node_id,
                        node_type=node.node_type.value,
                        text=text,
                        start=node.start if node.start is not None else 0.0,
                        end=node.end if node.end is not None and node.end != float('inf') else 0.0,
                        score=score_lookup.get(node_id, 0.0),
                        clip_id=node.clip_id if hasattr(node, 'clip_id') else None,
                        is_expanded=True,
                        expansion_source=expansion_source,
                    ),
                )
            )

        expanded_results.sort(key=lambda x: x[0])
        results.extend(result for _, result in expanded_results)

        # Extract subgraph
        subgraph = graph.get_subgraph(list(expanded_ids), include_edges=True)

        logger.info(
            f"Retrieved {len(seed_ids)} seed nodes and {len(expanded_results)} expanded nodes "
            f"using edges={[edge.value for edge in active_expansion_edge_types]} "
            f"(subgraph size={len(expanded_ids)})"
        )
        return results, subgraph
    
    def retrieve_by_timestamp(
        self,
        timestamp: float,
        graph: MultimodalGraph,
        window: float = 5.0
    ) -> List[RetrievalResult]:
        """
        Retrieve nodes around a specific timestamp.
        
        Args:
            timestamp: Time in seconds
            graph: The graph to search
            window: Time window in seconds
            
        Returns:
            List of nodes at the timestamp
        """
        results = []
        
        for node in graph.nodes.values():
            # Handle None values for start/end
            node_start = node.start if node.start is not None else 0
            node_end = node.end if node.end is not None else float('inf')
            
            if node_start - window <= timestamp <= node_end + window:
                # Use standardized function to get text representation
                text = get_node_text_for_embedding(node)
                
                # Handle None and infinity values
                node_start_safe = node.start if node.start is not None else 0.0
                node_end_safe = node.end if node.end is not None and node.end != float('inf') else (node_start_safe + 1.0)
                
                results.append(RetrievalResult(
                    node_id=node.id,
                    node_type=node.node_type.value,
                    text=text,
                    start=node_start_safe,
                    end=node_end_safe,
                    score=1.0 - abs(timestamp - node_start) / max(window, 1),
                    clip_id=node.clip_id if hasattr(node, 'clip_id') else None
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results


def retrieve_from_graph(
    query: str,
    graph_path: str,
    top_k: int = 10,
    hop_expansion: int = 2
) -> dict:
    """
    Convenience function for retrieval.
    
    Args:
        query: Search query
        graph_path: Path to graph.json
        top_k: Number of results
        hop_expansion: Hops for expansion
        
    Returns:
        Dictionary with results and subgraph
    """
    from ..graph.serialization import load_graph_json
    
    graph = load_graph_json(graph_path)
    retriever = GraphRetriever(top_k=top_k, hop_expansion=hop_expansion)
    
    results, subgraph = retriever.retrieve(query, graph)
    
    return {
        "results": [
            {
                "node_id": r.node_id,
                "node_type": r.node_type,
                "text": r.text,
                "start": r.start,
                "end": r.end,
                "score": r.score,
                "clip_id": r.clip_id,
                "is_expanded": r.is_expanded,
                "expansion_source": r.expansion_source,
            }
            for r in results
        ],
        "subgraph": subgraph.to_dict()
    }


