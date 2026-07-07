"""
Graph construction module.

Implements transcript-first semantic graph building:
1. Build topics + schema + discourse edges from transcript
2. Enrich with visual nodes and ALIGNED_TO edges
3. Entity canonicalization and linking
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
import uuid

from openai import OpenAI

from .models import (
    MultimodalGraph, TranscriptNode, VisualNode, EntityNode, TopicNode,
    Edge, EdgeType, DiscourseRelation, DiscourseSchema, NodeType
)
from ..cache.openai_cache import get_cache
from ..utils import get_node_text_for_embedding

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds a multimodal graph from video data."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        text_model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        temperature: float = 0.3,
        temporal_window: float = 5.0,
        build_topics: bool = True,
        build_entities: bool = True,
        cache_enabled: bool = True
    ):
        """
        Initialize the graph builder.
        
        Args:
            api_key: OpenAI API key
            text_model: Model for text analysis
            embedding_model: Model for embeddings
            temperature: Sampling temperature
            temporal_window: Seconds of overlap for ALIGNED_TO edges
            build_topics: Whether to build topic nodes
            build_entities: Whether to build entity nodes
            cache_enabled: Whether to cache API calls
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.client = OpenAI(api_key=api_key)
        self.text_model = text_model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.temporal_window = temporal_window
        self.build_topics = build_topics
        self.build_entities = build_entities
        self.cache = get_cache() if cache_enabled else None
    
    def build_graph(self, video_dir: str) -> MultimodalGraph:
        """
        Build the complete multimodal graph.
        
        This follows the transcript-first approach:
        1. Create transcript nodes from segments
        2. Analyze discourse structure and create topic nodes
        3. Add discourse relation edges
        4. Create visual nodes from clips
        5. Add ALIGNED_TO edges between transcript and visual nodes
        6. Extract and canonicalize entities
        7. Add entity edges
        
        Args:
            video_dir: Path to video directory
            
        Returns:
            MultimodalGraph object
        """
        video_dir = Path(video_dir)
        
        # Load metadata
        metadata_path = video_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {video_dir}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        video_id = metadata["metadata"]["video_id"]
        graph = MultimodalGraph(video_id=video_id)
        graph.metadata = {
            "title": metadata["metadata"].get("title", ""),
            "duration": metadata["metadata"].get("duration", 0),
            "build_log": []
        }
        
        logger.info(f"Building graph for video: {video_id}")
        
        # Step 1: Create transcript nodes
        transcript_nodes = self._create_transcript_nodes(video_dir)
        for node in transcript_nodes:
            graph.add_node(node)
        graph.metadata["build_log"].append(f"Created {len(transcript_nodes)} transcript nodes")
        
        # Step 2: Add temporal edges between transcript nodes
        self._add_temporal_edges(graph, transcript_nodes)
        
        # Step 3: Analyze discourse and create topics
        if self.build_topics and transcript_nodes:
            topics, discourse_edges = self._analyze_discourse(transcript_nodes)
            for topic in topics:
                graph.add_node(topic)
            for edge in discourse_edges:
                graph.add_edge(edge)
            graph.metadata["build_log"].append(f"Created {len(topics)} topics, {len(discourse_edges)} discourse edges")
        
        # Step 4: Create visual nodes
        visual_nodes = self._create_visual_nodes(video_dir)
        for node in visual_nodes:
            graph.add_node(node)
        graph.metadata["build_log"].append(f"Created {len(visual_nodes)} visual nodes")

        # Step 4b: Add temporal edges between visual nodes
        self._add_temporal_edges(graph, visual_nodes)
        
        # Step 5: Add ALIGNED_TO edges
        aligned_edges = self._add_aligned_edges(transcript_nodes, visual_nodes)
        for edge in aligned_edges:
            graph.add_edge(edge)
        graph.metadata["build_log"].append(f"Created {len(aligned_edges)} alignment edges")
        
        # Step 6: Extract and link entities
        if self.build_entities:
            entity_nodes, entity_edges = self._extract_entities(graph)
            for node in entity_nodes:
                graph.add_node(node)
            for edge in entity_edges:
                graph.add_edge(edge)
            graph.metadata["build_log"].append(f"Created {len(entity_nodes)} entities, {len(entity_edges)} entity edges")
        
        logger.info(f"Graph complete: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return graph
    
    def _create_transcript_nodes(self, video_dir: Path) -> List[TranscriptNode]:
        """Create transcript nodes from transcript.json."""
        # Try sentence segments first, fall back to regular segments
        sentences_path = video_dir / "transcript_sentences.json"
        transcript_path = video_dir / "transcript.json"
        
        if sentences_path.exists():
            with open(sentences_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            segments = data.get("sentence_segments", [])
        elif transcript_path.exists():
            with open(transcript_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            segments = data.get("segments", [])
        else:
            logger.warning("No transcript found")
            return []
        
        nodes = []
        for seg in segments:
            node = TranscriptNode(
                id=f"t_{seg.get('id', len(nodes)):04d}",
                node_type=NodeType.TRANSCRIPT,
                start=seg.get("start", 0),
                end=seg.get("end", 0),
                text=seg.get("text", "").strip(),
                sentence_id=seg.get("id")
            )
            if node.text:  # Skip empty segments
                nodes.append(node)
        
        logger.info(f"Created {len(nodes)} transcript nodes")
        return nodes
    
    def _add_temporal_edges(self, graph: MultimodalGraph, nodes: List):
        """Add TEMPORAL_NEXT edges between sequential nodes of the same modality."""
        sorted_nodes = sorted(nodes, key=lambda n: n.start)
        edge_prefix = "temp"

        if sorted_nodes:
            first_type = getattr(sorted_nodes[0], "node_type", None)
            if first_type == NodeType.VISUAL:
                edge_prefix = "temp_visual"
            elif first_type == NodeType.TRANSCRIPT:
                edge_prefix = "temp_transcript"
        
        for i in range(len(sorted_nodes) - 1):
            edge = Edge(
                id=f"{edge_prefix}_{i:04d}",
                source=sorted_nodes[i].id,
                target=sorted_nodes[i + 1].id,
                edge_type=EdgeType.TEMPORAL_NEXT
            )
            graph.add_edge(edge)
    
    def _analyze_discourse(
        self,
        transcript_nodes: List[TranscriptNode]
    ) -> Tuple[List[TopicNode], List[Edge]]:
        """
        Analyze discourse structure using LLM.
        
        Returns:
            Tuple of (topic_nodes, discourse_edges)
        """
        # Windowed analysis: never pass more than WINDOW sentences per call (a flat
        # single pass is unbounded on long transcripts). Short transcripts = one call
        # with the original prompt (cache-key stable).
        WINDOW = 40
        all_topics_data: List[dict] = []
        all_relations_data: List[dict] = []
        for wi in range(0, len(transcript_nodes), WINDOW):
            window_nodes = transcript_nodes[wi:wi + WINDOW]
            data = self._analyze_discourse_window(window_nodes)
            for t in data.get("topics", []):
                if len(transcript_nodes) > WINDOW:
                    t["id"] = f"w{wi // WINDOW}_{t.get('id', 'topic')}"
                all_topics_data.append(t)
            all_relations_data.extend(data.get("discourse_relations", []))
        data = {"topics": all_topics_data, "discourse_relations": all_relations_data}

        # Create topic nodes
        topics = []
        node_to_topic = {}

        for topic_data in data.get("topics", []):
            # Find time range from member segments
            member_ids = topic_data.get("member_segments", [])
            member_nodes = [n for n in transcript_nodes if n.id in member_ids]

            if not member_nodes:
                continue

            start = min(n.start for n in member_nodes)
            end = max(n.end for n in member_nodes)

            schema_str = topic_data.get("schema", "informative")
            try:
                schema = DiscourseSchema(schema_str.lower())
            except ValueError:
                schema = DiscourseSchema.INFORMATIVE

            topic = TopicNode(
                id=topic_data.get("id", f"topic_{len(topics)}"),
                node_type=NodeType.TOPIC,
                start=start,
                end=end,
                title=topic_data.get("title", ""),
                description=topic_data.get("description"),
                schema=schema,
                keywords=topic_data.get("keywords", []),
                member_nodes=member_ids
            )
            topics.append(topic)

            # Update transcript nodes with topic_id and schema
            for node in member_nodes:
                node.topic_id = topic.id
                node.schema = schema
                node_to_topic[node.id] = topic.id

        # Create discourse edges
        edges = []
        valid_relations = {r.value for r in DiscourseRelation}

        for rel_data in data.get("discourse_relations", []):
            source = rel_data.get("source")
            target = rel_data.get("target")
            relation_str = rel_data.get("relation", "").upper()

            if source and target and relation_str in valid_relations:
                edge = Edge(
                    id=f"disc_{len(edges):04d}",
                    source=source,
                    target=target,
                    edge_type=EdgeType.DISCOURSE_RELATION,
                    relation=DiscourseRelation(relation_str)
                )
                edges.append(edge)

        # Add BELONGS_TO_TOPIC edges
        for node_id, topic_id in node_to_topic.items():
            edge = Edge(
                id=f"topic_edge_{len(edges):04d}",
                source=node_id,
                target=topic_id,
                edge_type=EdgeType.BELONGS_TO_TOPIC
            )
            edges.append(edge)

        logger.info(f"Analyzed {len(topics)} topics, {len(edges)} discourse edges")
        return topics, edges

    def _analyze_discourse_window(self, transcript_nodes: List[TranscriptNode]) -> dict:
        """One bounded discourse-analysis call over a window of transcript sentences."""
        text_with_ids = "\n".join([
            f"[{node.id}] {node.text}"
            for node in transcript_nodes
        ])

        prompt = f"""Analyze the following transcript and identify:
1. Topic segments (groups of related sentences)
2. The discourse schema of each topic (narrative, descriptive, informative, instructional, argumentative)
3. Discourse relations between segments

Transcript:
{text_with_ids}

Return a JSON object with:
{{
    "topics": [
        {{
            "id": "topic_0",
            "title": "Short topic title",
            "description": "Brief description",
            "schema": "informative|narrative|descriptive|instructional|argumentative",
            "keywords": ["key", "words"],
            "member_segments": ["t_0000", "t_0001", ...]
        }}
    ],
    "discourse_relations": [
        {{
            "source": "t_0001",
            "target": "t_0002",
            "relation": "EXPLAINS|EXAMPLE_OF|SUPPORT|COUNTER|CAUSES|STEP_BEFORE|ELABORATES|etc."
        }}
    ]
}}

Be conservative with discourse relations - only include clear, strong relations."""

        messages = [{"role": "user", "content": prompt}]
        params = {"temperature": self.temperature, "max_tokens": 16384, "seed": 0}
        
        # Check cache
        if self.cache:
            cached = self.cache.get(self.text_model, messages, params)
            if cached:
                response_text = cached.get("text", "{}")
            else:
                response = self.client.chat.completions.create(
                    model=self.text_model,
                    messages=messages,
                    **params,
                    response_format={"type": "json_object"}
                )
                response_text = response.choices[0].message.content
                self.cache.set(self.text_model, messages, params, {"text": response_text})
        else:
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=messages,
                **params,
                response_format={"type": "json_object"}
            )
            response_text = response.choices[0].message.content
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse discourse analysis window")
            return {}

    def _create_visual_nodes(self, video_dir: Path) -> List[VisualNode]:
        """Create visual nodes from visual.json."""
        visual_path = video_dir / "visual.json"
        
        if not visual_path.exists():
            logger.warning("No visual.json found")
            return []
        
        with open(visual_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        nodes = []
        for analysis in data.get("analyses", []):
            node = VisualNode(
                id=f"v_{analysis['clip_id']}",
                node_type=NodeType.VISUAL,
                start=analysis.get("start", 0),
                end=analysis.get("end", 0),
                clip_id=analysis["clip_id"],
                visual_description=analysis.get("visual_description", ""),
                ocr_text=analysis.get("ocr_text"),
                detected_entities=analysis.get("detected_entities", []),
                state_change_from_previous=analysis.get("state_change_from_previous", ""),
                scene_type=analysis.get("scene_type"),
                keyframes=analysis.get("keyframes_analyzed", [])
            )
            nodes.append(node)
        logger.info(f"Created {len(nodes)} visual nodes")
        return nodes
    
    def _add_aligned_edges(
        self,
        transcript_nodes: List[TranscriptNode],
        visual_nodes: List[VisualNode]
    ) -> List[Edge]:
        """Create ALIGNED_TO edges between temporally overlapping nodes."""
        edges = []
        
        for t_node in transcript_nodes:
            for v_node in visual_nodes:
                # Check for temporal overlap with window
                t_start = t_node.start - self.temporal_window
                t_end = t_node.end + self.temporal_window
                
                if t_start <= v_node.end and t_end >= v_node.start:
                    # Calculate overlap for weight
                    overlap_start = max(t_node.start, v_node.start)
                    overlap_end = min(t_node.end, v_node.end)
                    overlap = max(0, overlap_end - overlap_start)
                    
                    t_duration = t_node.end - t_node.start
                    weight = overlap / max(t_duration, 0.1)
                    
                    edge = Edge(
                        id=f"align_{len(edges):04d}",
                        source=t_node.id,
                        target=v_node.id,
                        edge_type=EdgeType.ALIGNED_TO,
                        weight=min(weight, 1.0)
                    )
                    edges.append(edge)
        
        return edges
    
    def _extract_entities(
        self,
        graph: MultimodalGraph
    ) -> Tuple[List[EntityNode], List[Edge]]:
        """
        Extract and canonicalize entities from the graph.
        
        Uses conservative entity linking to avoid over-merging.
        """
        # Collect all text for entity extraction
        context_blocks = []
        node_texts = {}
        
        for node in graph.nodes.values():
            if node.node_type == NodeType.TRANSCRIPT:
                text = node.text
                context_blocks.append(
                    f"[Transcript {node.id} | {node.start:.1f}s-{node.end:.1f}s]\n"
                    f"Text: {text}"
                )
                node_texts[node.id] = text
            elif node.node_type == NodeType.VISUAL:
                visual_lines = []
                if node.visual_description:
                    visual_lines.append(f"Visual description: {node.visual_description}")
                if getattr(node, "state_change_from_previous", None):
                    visual_lines.append(
                        "State change from previous clip: "
                        f"{node.state_change_from_previous}"
                    )
                if node.detected_entities:
                    visual_lines.append(
                        f"Detected entities: {', '.join(node.detected_entities)}"
                    )
                if node.ocr_text:
                    visual_lines.append(f"OCR text: {node.ocr_text}")

                if visual_lines:
                    context_blocks.append(
                        f"[Visual {node.id} | {node.start:.1f}s-{node.end:.1f}s]\n"
                        + "\n".join(visual_lines)
                    )

                node_text_parts = []
                if node.visual_description:
                    node_text_parts.append(node.visual_description)
                if getattr(node, "state_change_from_previous", None):
                    node_text_parts.append(node.state_change_from_previous)
                if node.detected_entities:
                    node_text_parts.append(", ".join(node.detected_entities))
                if node.ocr_text:
                    node_text_parts.append(node.ocr_text)
                node_texts[node.id] = " ".join(node_text_parts)

        def _extract_window(window_context: str) -> list:
            prompt = f"""Extract named entities from the structured video context below. The context contains transcript snippets and visual observations from the same video. For each entity, provide:
- Canonical name (shortest common form)
- Entity type (person, organization, product, concept, location, event, other)
- Aliases (other forms used in the text)

Context:
{window_context}

Return JSON:
{{
    "entities": [
        {{
            "name": "canonical name",
            "type": "person|organization|product|concept|location|event|other",
            "aliases": ["alias1", "alias2"]
        }}
    ]
}}

Be conservative - only extract clear named entities. Prefer shorter canonical names.
Use transcript text, visual descriptions, detected entities, and OCR text as evidence."""
            messages = [{"role": "user", "content": prompt}]
            params = {"temperature": self.temperature, "max_tokens": 4096, "seed": 0}
            response_text = None
            if self.cache:
                cached = self.cache.get(self.text_model, messages, params)
                if cached:
                    response_text = cached.get("text", "{}")
            if response_text is None:
                response = self.client.chat.completions.create(
                    model=self.text_model, messages=messages, **params,
                    response_format={"type": "json_object"}
                )
                response_text = response.choices[0].message.content
                if self.cache:
                    self.cache.set(self.text_model, messages, params, {"text": response_text})
            try:
                return json.loads(response_text).get("entities", [])
            except json.JSONDecodeError:
                logger.warning("Failed to parse entity extraction window")
                return []

        # Windowed extraction: never pass more than ~10k chars of context per call (a flat
        # single capped call silently ignored everything beyond the cap on long videos).
        # Windows are merged by casefold canonical name, unioning aliases.
        windows, cur, cur_len = [], [], 0
        for block in context_blocks:
            if cur and cur_len + len(block) > 10000:
                windows.append("\n\n".join(cur)); cur, cur_len = [], 0
            cur.append(block); cur_len += len(block) + 2
        if cur:
            windows.append("\n\n".join(cur))

        merged: Dict[str, dict] = {}
        for w in windows:
            for ent in _extract_window(w):
                name = (ent.get("name") or "").strip()
                if not name:
                    continue
                key = name.lower()
                if key in merged:
                    merged[key]["aliases"] = list({*merged[key].get("aliases", []),
                                                   *(ent.get("aliases") or [])})
                else:
                    merged[key] = {"name": name, "type": ent.get("type", "other"),
                                   "aliases": ent.get("aliases", []) or []}

        # Create entity nodes
        entity_nodes = []
        entity_edges = []

        for entity_data in merged.values():
            name = entity_data.get("name", "")
            if not name:
                continue
            
            entity = EntityNode(
                id=f"e_{uuid.uuid4().hex[:8]}",
                node_type=NodeType.ENTITY,
                start=0,
                end=float('inf'),
                name=name,
                entity_type=entity_data.get("type", "other"),
                aliases=entity_data.get("aliases", [])
            )
            
            # Find mentions in nodes
            all_forms = [name.lower()] + [a.lower() for a in entity.aliases]
            mentions = []
            
            for node_id, text in node_texts.items():
                text_lower = text.lower()
                for form in all_forms:
                    if form in text_lower:
                        mentions.append(node_id)
                        break
            
            entity.mentions = list(set(mentions))
            
            if entity.mentions:  # Only add entities that are actually mentioned
                entity_nodes.append(entity)
                
                # Create CONTAINS_ENTITY edges
                for mention_id in entity.mentions:
                    edge = Edge(
                        id=f"ent_{len(entity_edges):04d}",
                        source=mention_id,
                        target=entity.id,
                        edge_type=EdgeType.CONTAINS_ENTITY
                    )
                    entity_edges.append(edge)
        
        logger.info(f"Extracted {len(entity_nodes)} entities")
        return entity_nodes, entity_edges
    
    def save_graph(self, graph: MultimodalGraph, output_path: Path):
        """Save graph to JSON file."""
        from .serialization import SafeJSONEncoder
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph.to_dict(), f, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)
        
        logger.info(f"Graph saved to {output_path}")
    
    def export_graphml(self, graph: MultimodalGraph, output_path: Path):
        """Export graph to GraphML format."""
        try:
            import networkx as nx
            from enum import Enum
            
            G = nx.DiGraph()
            
            # Add nodes
            for node_id, node in graph.nodes.items():
                attrs = node.to_dict()
                # Convert complex types to strings for GraphML compatibility
                cleaned_attrs = {}
                for k, v in attrs.items():
                    if isinstance(v, Enum):
                        cleaned_attrs[k] = v.value
                    elif isinstance(v, (list, dict)):
                        cleaned_attrs[k] = json.dumps(v)
                    elif v is None:
                        cleaned_attrs[k] = ""
                    elif isinstance(v, float) and (v == float('inf') or v == float('-inf')):
                        cleaned_attrs[k] = -1
                    else:
                        cleaned_attrs[k] = v
                G.add_node(node_id, **cleaned_attrs)
            
            # Add edges
            for edge_id, edge in graph.edges.items():
                attrs = edge.to_dict()
                cleaned_attrs = {}
                for k, v in attrs.items():
                    if isinstance(v, Enum):
                        cleaned_attrs[k] = v.value
                    elif isinstance(v, (list, dict)):
                        cleaned_attrs[k] = json.dumps(v)
                    elif v is None:
                        cleaned_attrs[k] = ""
                    elif isinstance(v, float) and (v == float('inf') or v == float('-inf')):
                        cleaned_attrs[k] = -1
                    else:
                        cleaned_attrs[k] = v
                G.add_edge(edge.source, edge.target, **cleaned_attrs)
            
            nx.write_graphml(G, output_path)
            logger.info(f"GraphML exported to {output_path}")
        except ImportError:
            logger.warning("networkx not installed, skipping GraphML export")
        except Exception as e:
            logger.warning(f"GraphML export failed (non-fatal): {e}")


def build_video_graph(
    video_dir: str,
    output_path: Optional[str] = None,
    compute_embeddings: bool = True,
    max_parallel_embeddings: Optional[int] = None,
    progress_callback=None,
    config: Optional[dict] = None,
) -> dict:
    """
    Convenience function to build a graph from a video directory.
    
    Args:
        video_dir: Path to video directory
        output_path: Path to save graph (defaults to video_dir/graph.json)
        compute_embeddings: Whether to pre-compute embeddings for retrieval
        max_parallel_embeddings: Maximum parallel embedding API calls
        progress_callback: Optional callback for progress updates
        config: Optional VideoGraph configuration dictionary
        
    Returns:
        Graph dictionary
    """
    video_dir = Path(video_dir)
    output_path = Path(output_path) if output_path else video_dir / "graph.json"

    config = config or {}
    openai_config = config.get("openai", {})
    graph_config = config.get("graph", {})
    processing_config = config.get("processing", {})

    text_model = graph_config.get("text_model", openai_config.get("text_model", "gpt-4o"))
    embedding_model = graph_config.get(
        "embedding_model",
        openai_config.get("embedding_model", "text-embedding-3-small"),
    )
    graph_temperature = float(graph_config.get("temperature", 0.3))
    temporal_window = float(graph_config.get("temporal_window", 5.0))
    build_topics = bool(graph_config.get("build_topic_nodes", True))
    build_entities = bool(graph_config.get("build_entity_nodes", True))
    if max_parallel_embeddings is None:
        max_parallel_embeddings = int(processing_config.get("max_parallel_embeddings", 10))
    
    builder = GraphBuilder(
        text_model=text_model,
        embedding_model=embedding_model,
        temperature=graph_temperature,
        temporal_window=temporal_window,
        build_topics=build_topics,
        build_entities=build_entities,
    )
    graph = builder.build_graph(video_dir)
    builder.save_graph(graph, output_path)
    
    # Also export GraphML
    graphml_path = output_path.with_suffix('.graphml')
    builder.export_graphml(graph, graphml_path)
    
    # Pre-compute embeddings for fast retrieval
    if compute_embeddings:
        embeddings_path = video_dir / "embeddings.json"
        compute_graph_embeddings(
            graph,
            embeddings_path,
            embedding_model=embedding_model,
            max_parallel=max_parallel_embeddings,
            progress_callback=progress_callback
        )

        from ..retrieval.graph_retrieval import GraphRetriever

        GraphRetriever(
            embedding_model=embedding_model,
        ).prepare_visual_channel_embeddings(graph, embeddings_path)
    
    return graph.to_dict()


def compute_graph_embeddings(
    graph,
    output_path: Path,
    embedding_model: str = "text-embedding-3-small",
    max_parallel: int = 10,
    progress_callback=None
):
    """
    Pre-compute embeddings for all nodes in the graph.
    
    Args:
        graph: MultimodalGraph object
        output_path: Path to save embeddings JSON
        embedding_model: OpenAI embedding model
        max_parallel: Maximum parallel API calls
        progress_callback: Optional callback for progress updates
    """
    import os
    from openai import OpenAI
    from ..processing.parallel import ParallelProcessor
    from ..cache.openai_cache import get_cache
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found, skipping embedding computation")
        return
    
    client = OpenAI(api_key=api_key)
    cache = get_cache()
    
    # Collect texts to embed using standardized function
    embed_tasks = []
    for node_id, node in graph.nodes.items():
        text = get_node_text_for_embedding(node)
        
        if text.strip():
            embed_tasks.append({"node_id": node_id, "text": text[:8000]})
    
    if not embed_tasks:
        logger.warning("No nodes to embed")
        return
    
    logger.info(f"Computing embeddings for {len(embed_tasks)} nodes...")
    
    def get_embedding(task):
        text = task["text"]
        node_id = task["node_id"]
        
        # Check cache
        cache_key = {"text": text[:1000], "model": embedding_model}
        if cache:
            cached = cache.get(
                embedding_model,
                [{"role": "embed", "content": text[:100]}],
                cache_key
            )
            if cached:
                emb = cached.get("embedding", [])
                # Discard corrupt cached embeddings (historic truncated writes)
                if len(emb) in (1536, 3072):
                    return {"node_id": node_id, "embedding": emb}
        
        try:
            response = client.embeddings.create(
                model=embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Cache result
            if cache:
                cache.set(
                    embedding_model,
                    [{"role": "embed", "content": text[:100]}],
                    cache_key,
                    {"embedding": embedding}
                )
            
            return {"node_id": node_id, "embedding": embedding}
        except Exception as e:
            logger.warning(f"Failed to embed node {node_id}: {e}")
            return None
    
    # Process embeddings in parallel
    processor = ParallelProcessor(
        max_workers=max_parallel,
        rate_limit_rpm=3000,  # Embedding API has higher rate limits
        progress_callback=progress_callback
    )
    
    results = processor.process_parallel(
        embed_tasks,
        get_embedding,
        stage_name="embedding_computation",
        item_name="node"
    )
    
    # Build embeddings dict
    embeddings = {}
    for result in results:
        if result:
            embeddings[result["node_id"]] = result["embedding"]
    
    # Save embeddings
    with open(output_path, 'w') as f:
        json.dump(embeddings, f)
    
    logger.info(f"Saved embeddings to {output_path} ({len(embeddings)} nodes)")


