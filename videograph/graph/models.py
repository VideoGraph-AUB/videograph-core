"""
Graph data models for the multimodal video graph.

Node Types:
- TranscriptNode: Text segments from the video transcript
- VisualNode: Visual clips with descriptions and OCR
- EntityNode: Canonical entities (people, objects, concepts)
- TopicNode: Topic segments grouping related content

Edge Types:
- TEMPORAL_NEXT: Sequential ordering
- ALIGNED_TO: Temporal alignment between modalities
- DISCOURSE_RELATION: Schema-driven relations
- SAME_ENTITY / COREF: Entity coreference
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any
import uuid


class DiscourseSchema(str, Enum):
    """High-level discourse schemas for video content."""
    NARRATIVE = "narrative"
    DESCRIPTIVE = "descriptive"
    INFORMATIVE = "informative"
    INSTRUCTIONAL = "instructional"
    ARGUMENTATIVE = "argumentative"


class DiscourseRelation(str, Enum):
    """Discourse relation types between nodes."""
    # Rhetorical relations
    SUPPORT = "SUPPORT"
    COUNTER = "COUNTER"
    EXPLAINS = "EXPLAINS"
    DEFINES = "DEFINES"
    EXAMPLE_OF = "EXAMPLE_OF"
    ELABORATES = "ELABORATES"
    SUMMARIZES = "SUMMARIZES"
    CONTRASTS = "CONTRASTS"
    
    # Temporal/causal relations
    STEP_BEFORE = "STEP_BEFORE"
    STEP_AFTER = "STEP_AFTER"
    CAUSES = "CAUSES"
    RESULTS_FROM = "RESULTS_FROM"
    ENABLES = "ENABLES"
    
    # Structural relations
    INTRODUCES = "INTRODUCES"
    CONCLUDES = "CONCLUDES"
    TRANSITIONS = "TRANSITIONS"
    
    # Content relations
    REFERENCES = "REFERENCES"
    QUOTES = "QUOTES"
    CITES = "CITES"


class EdgeType(str, Enum):
    """Edge types in the graph."""
    TEMPORAL_NEXT = "TEMPORAL_NEXT"
    ALIGNED_TO = "ALIGNED_TO"
    DISCOURSE_RELATION = "DISCOURSE_RELATION"
    SAME_ENTITY = "SAME_ENTITY"
    COREF = "COREF"
    BELONGS_TO_TOPIC = "BELONGS_TO_TOPIC"
    CONTAINS_ENTITY = "CONTAINS_ENTITY"
    VISUALLY_DEPICTS = "VISUALLY_DEPICTS"


class NodeType(str, Enum):
    """Node types in the graph."""
    TRANSCRIPT = "TranscriptNode"
    VISUAL = "VisualNode"
    ENTITY = "EntityNode"
    TOPIC = "TopicNode"


@dataclass
class BaseNode:
    """Base class for all nodes."""
    id: str
    node_type: NodeType
    start: float  # seconds
    end: float  # seconds
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TranscriptNode(BaseNode):
    """Node representing a transcript segment."""
    text: str
    topic_id: Optional[str] = None
    schema: Optional[DiscourseSchema] = None
    speaker: Optional[str] = None
    embedding: Optional[List[float]] = None
    sentence_id: Optional[int] = None
    
    def __post_init__(self):
        self.node_type = NodeType.TRANSCRIPT
        if not self.id:
            self.id = f"t_{uuid.uuid4().hex[:8]}"


@dataclass
class VisualNode(BaseNode):
    """Node representing a visual clip."""
    clip_id: str
    visual_description: str
    state_change_from_previous: Optional[str] = None
    ocr_text: Optional[str] = None
    detected_entities: List[str] = field(default_factory=list)
    scene_type: Optional[str] = None
    keyframes: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        self.node_type = NodeType.VISUAL
        if not self.id:
            self.id = f"v_{self.clip_id}"


@dataclass
class EntityNode(BaseNode):
    """Node representing a canonical entity."""
    name: str
    entity_type: str  # person, object, concept, organization, etc.
    aliases: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)  # node IDs that mention this entity
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        self.node_type = NodeType.ENTITY
        if not self.id:
            self.id = f"e_{uuid.uuid4().hex[:8]}"
        # Entities span the entire video by default
        if self.start == 0 and self.end == 0:
            self.start = 0
            self.end = float('inf')


@dataclass
class TopicNode(BaseNode):
    """Node representing a topic segment."""
    title: str
    description: Optional[str] = None
    schema: Optional[DiscourseSchema] = None
    keywords: List[str] = field(default_factory=list)
    member_nodes: List[str] = field(default_factory=list)  # transcript node IDs
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        self.node_type = NodeType.TOPIC
        if not self.id:
            self.id = f"topic_{uuid.uuid4().hex[:8]}"


@dataclass
class Edge:
    """Edge connecting two nodes."""
    id: str
    source: str  # node ID
    target: str  # node ID
    edge_type: EdgeType
    relation: Optional[DiscourseRelation] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"edge_{uuid.uuid4().hex[:8]}"
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["edge_type"] = self.edge_type.value if isinstance(self.edge_type, EdgeType) else self.edge_type
        if self.relation:
            d["relation"] = self.relation.value if isinstance(self.relation, DiscourseRelation) else self.relation
        return d


@dataclass
class MultimodalGraph:
    """The complete multimodal video graph."""
    video_id: str
    nodes: Dict[str, BaseNode] = field(default_factory=dict)
    edges: Dict[str, Edge] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: BaseNode):
        """Add a node to the graph."""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: Edge):
        """Add an edge to the graph."""
        self.edges[edge.id] = edge
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[BaseNode]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes.values() if n.node_type == node_type]
    
    def get_edges_by_type(self, edge_type: EdgeType) -> List[Edge]:
        """Get all edges of a specific type."""
        return [e for e in self.edges.values() if e.edge_type == edge_type]
    
    def get_neighbors(self, node_id: str, edge_types: Optional[List[EdgeType]] = None) -> List[str]:
        """Get neighboring node IDs."""
        neighbors = []
        for edge in self.edges.values():
            if edge_types and edge.edge_type not in edge_types:
                continue
            if edge.source == node_id:
                neighbors.append(edge.target)
            elif edge.target == node_id:
                neighbors.append(edge.source)
        return list(set(neighbors))
    
    def get_subgraph(
        self,
        node_ids: List[str],
        include_edges: bool = True
    ) -> 'MultimodalGraph':
        """Extract a subgraph containing the specified nodes."""
        subgraph = MultimodalGraph(video_id=self.video_id)
        
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.nodes[node_id] = self.nodes[node_id]
        
        if include_edges:
            for edge_id, edge in self.edges.items():
                if edge.source in node_ids and edge.target in node_ids:
                    subgraph.edges[edge_id] = edge
        
        return subgraph
    
    def expand_subgraph(
        self,
        node_ids: List[str],
        hops: int = 1,
        edge_types: Optional[List[EdgeType]] = None
    ) -> List[str]:
        """Expand a set of nodes by k hops."""
        current = set(node_ids)
        
        for _ in range(hops):
            new_nodes = set()
            for node_id in current:
                neighbors = self.get_neighbors(node_id, edge_types)
                new_nodes.update(neighbors)
            current.update(new_nodes)
        
        return list(current)
    
    def get_node_at_timestamp(self, timestamp: float, node_type: Optional[NodeType] = None) -> List[BaseNode]:
        """Get nodes that span a given timestamp."""
        result = []
        for node in self.nodes.values():
            if node_type and node.node_type != node_type:
                continue
            if node.start <= timestamp <= node.end:
                result.append(node)
        return result
    
    def to_dict(self) -> dict:
        """Serialize the graph to a dictionary."""
        nodes_list = []
        for node in self.nodes.values():
            node_dict = node.to_dict()
            # Convert enums to strings
            node_dict["node_type"] = node.node_type.value if isinstance(node.node_type, NodeType) else node.node_type
            if hasattr(node, 'schema') and node.schema:
                node_dict["schema"] = node.schema.value if isinstance(node.schema, DiscourseSchema) else node.schema
            nodes_list.append(node_dict)
        
        edges_list = [edge.to_dict() for edge in self.edges.values()]
        
        return {
            "video_id": self.video_id,
            "nodes": nodes_list,
            "edges": edges_list,
            "metadata": self.metadata,
            "stats": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "nodes_by_type": {
                    nt.value: len(self.get_nodes_by_type(nt))
                    for nt in NodeType
                },
                "edges_by_type": {
                    et.value: len(self.get_edges_by_type(et))
                    for et in EdgeType
                }
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MultimodalGraph':
        """Deserialize a graph from a dictionary."""
        graph = cls(video_id=data.get("video_id", ""))
        graph.metadata = data.get("metadata", {})
        
        # Load nodes
        for node_data in data.get("nodes", []):
            node_type = NodeType(node_data.get("node_type", "TranscriptNode"))
            
            if node_type == NodeType.TRANSCRIPT:
                node = TranscriptNode(
                    id=node_data["id"],
                    node_type=node_type,
                    start=node_data.get("start", 0),
                    end=node_data.get("end", 0),
                    text=node_data.get("text", ""),
                    topic_id=node_data.get("topic_id"),
                    schema=DiscourseSchema(node_data["schema"]) if node_data.get("schema") else None,
                    speaker=node_data.get("speaker"),
                    embedding=node_data.get("embedding"),
                    sentence_id=node_data.get("sentence_id")
                )
            elif node_type == NodeType.VISUAL:
                node = VisualNode(
                    id=node_data["id"],
                    node_type=node_type,
                    start=node_data.get("start", 0),
                    end=node_data.get("end", 0),
                    clip_id=node_data.get("clip_id", ""),
                    visual_description=node_data.get("visual_description", ""),
                    state_change_from_previous=node_data.get("state_change_from_previous"),
                    ocr_text=node_data.get("ocr_text"),
                    detected_entities=node_data.get("detected_entities", []),
                    scene_type=node_data.get("scene_type"),
                    keyframes=node_data.get("keyframes", []),
                    embedding=node_data.get("embedding")
                )
            elif node_type == NodeType.ENTITY:
                node = EntityNode(
                    id=node_data["id"],
                    node_type=node_type,
                    start=node_data.get("start", 0),
                    end=node_data.get("end", float('inf')),
                    name=node_data.get("name", ""),
                    entity_type=node_data.get("entity_type", "unknown"),
                    aliases=node_data.get("aliases", []),
                    mentions=node_data.get("mentions", []),
                    embedding=node_data.get("embedding")
                )
            elif node_type == NodeType.TOPIC:
                node = TopicNode(
                    id=node_data["id"],
                    node_type=node_type,
                    start=node_data.get("start", 0),
                    end=node_data.get("end", 0),
                    title=node_data.get("title", ""),
                    description=node_data.get("description"),
                    schema=DiscourseSchema(node_data["schema"]) if node_data.get("schema") else None,
                    keywords=node_data.get("keywords", []),
                    member_nodes=node_data.get("member_nodes", []),
                    embedding=node_data.get("embedding")
                )
            else:
                continue
            
            graph.nodes[node.id] = node
        
        # Load edges
        for edge_data in data.get("edges", []):
            edge = Edge(
                id=edge_data["id"],
                source=edge_data["source"],
                target=edge_data["target"],
                edge_type=EdgeType(edge_data["edge_type"]),
                relation=DiscourseRelation(edge_data["relation"]) if edge_data.get("relation") else None,
                weight=edge_data.get("weight", 1.0),
                metadata=edge_data.get("metadata", {})
            )
            graph.edges[edge.id] = edge
        
        return graph


