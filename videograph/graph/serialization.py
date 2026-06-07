"""
Graph serialization utilities.

Supports JSON and GraphML formats with full timestamp preservation.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from .models import MultimodalGraph

logger = logging.getLogger(__name__)


class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles infinity and other special values."""
    
    def default(self, obj):
        if isinstance(obj, float):
            if obj == float('inf') or obj == float('-inf'):
                return None
        if hasattr(obj, 'value'):  # Handle enums
            return obj.value
        return str(obj)
    
    def encode(self, obj):
        return super().encode(self._sanitize(obj))
    
    def _sanitize(self, obj):
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize(item) for item in obj]
        elif isinstance(obj, float):
            if obj == float('inf') or obj == float('-inf'):
                return None
        return obj


def save_graph_json(graph: MultimodalGraph, path: Path):
    """Save graph to JSON format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(graph.to_dict(), f, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)
    
    logger.info(f"Graph saved to {path}")


def load_graph_json(path: Path) -> MultimodalGraph:
    """Load graph from JSON format."""
    path = Path(path)
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Handle JavaScript Infinity values that may be in older files
    content = content.replace(': Infinity', ': null')
    content = content.replace(': -Infinity', ': null')
    data = json.loads(content)
    
    return MultimodalGraph.from_dict(data)


def save_graph_graphml(graph: MultimodalGraph, path: Path):
    """Export graph to GraphML format for visualization tools."""
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not installed, cannot export GraphML")
        return
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    G = nx.DiGraph()
    
    # Add nodes with attributes
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
                cleaned_attrs[k] = -1  # Use -1 to represent infinity
            else:
                cleaned_attrs[k] = v
        G.add_node(node_id, **cleaned_attrs)
    
    # Add edges with attributes
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
    
    nx.write_graphml(G, path)
    logger.info(f"GraphML exported to {path}")


def load_graph_graphml(path: Path) -> MultimodalGraph:
    """Load graph from GraphML format."""
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx required for GraphML support")
    
    path = Path(path)
    G = nx.read_graphml(path)
    
    # Convert back to MultimodalGraph
    # Note: This loses some type information, prefer JSON for full fidelity
    data = {
        "video_id": G.graph.get("video_id", ""),
        "nodes": [],
        "edges": [],
        "metadata": {}
    }
    
    for node_id in G.nodes():
        node_data = dict(G.nodes[node_id])
        node_data["id"] = node_id
        # Parse JSON strings back to lists/dicts
        for k, v in node_data.items():
            if isinstance(v, str):
                try:
                    node_data[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    pass
        data["nodes"].append(node_data)
    
    for source, target, edge_data in G.edges(data=True):
        edge_dict = dict(edge_data)
        edge_dict["source"] = source
        edge_dict["target"] = target
        # Parse JSON strings
        for k, v in edge_dict.items():
            if isinstance(v, str):
                try:
                    edge_dict[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    pass
        data["edges"].append(edge_dict)
    
    return MultimodalGraph.from_dict(data)


def get_graph_statistics(graph: MultimodalGraph) -> dict:
    """Compute statistics about the graph."""
    from .models import NodeType, EdgeType
    
    stats = {
        "total_nodes": len(graph.nodes),
        "total_edges": len(graph.edges),
        "nodes_by_type": {},
        "edges_by_type": {},
        "avg_degree": 0,
        "density": 0,
        "time_span": {"start": float('inf'), "end": 0}
    }
    
    # Count by type
    for node_type in NodeType:
        count = len(graph.get_nodes_by_type(node_type))
        if count > 0:
            stats["nodes_by_type"][node_type.value] = count
    
    for edge_type in EdgeType:
        count = len(graph.get_edges_by_type(edge_type))
        if count > 0:
            stats["edges_by_type"][edge_type.value] = count
    
    # Compute degree
    if graph.nodes:
        degree_sum = 0
        for node_id in graph.nodes:
            degree = len(graph.get_neighbors(node_id))
            degree_sum += degree
        stats["avg_degree"] = round(degree_sum / len(graph.nodes), 2)
    
    # Compute density
    n = len(graph.nodes)
    if n > 1:
        max_edges = n * (n - 1)  # Directed graph
        stats["density"] = round(len(graph.edges) / max_edges, 4)
    
    # Compute time span
    for node in graph.nodes.values():
        if node.start < stats["time_span"]["start"] and node.start != float('-inf'):
            stats["time_span"]["start"] = node.start
        if node.end > stats["time_span"]["end"] and node.end != float('inf'):
            stats["time_span"]["end"] = node.end
    
    if stats["time_span"]["start"] == float('inf'):
        stats["time_span"]["start"] = 0
    
    return stats


