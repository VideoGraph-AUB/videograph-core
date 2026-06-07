# Graph Building Methodology

This document describes the graph-building behavior implemented in
`videograph/graph/builder.py`.

VideoGraph builds a persistent multimodal graph for each processed video. The
graph stores transcript evidence, visual evidence, topic structure, entity
mentions, temporal order, and cross-modal alignment. Retrieval can then reuse
this graph across many questions without rebuilding the video representation.

## Inputs

Graph construction expects a processed video directory containing some or all of
the following artifacts:

```text
metadata.json
transcript_sentences.json
transcript.json
visual.json
```

`metadata.json` is required because it provides the video id and basic video
metadata. Transcript and visual artifacts are optional in the sense that the
builder can continue if one modality is missing, but the resulting graph will
only contain the available evidence.

The core package supports two ingestion paths before graph construction:

- YouTube/demo ingestion through `videograph.video.io.process_youtube_video`.
- Local adaptive ingestion through
  `videograph.video.adaptive_ingest.process_local_video_adaptive`, used by the
  evaluation harness for already-downloaded benchmark videos.

Both paths produce artifacts that can be consumed by the same graph builder.

## Construction Steps

`GraphBuilder.build_graph()` follows this sequence:

1. Load `metadata.json` and initialize a `MultimodalGraph`.
2. Create transcript nodes from `transcript_sentences.json`, falling back to
   `transcript.json` if sentence-level segments are unavailable.
3. Add `TEMPORAL_NEXT` edges between consecutive transcript nodes.
4. Analyze transcript discourse with the configured text model and create topic
   nodes, discourse-relation edges, and topic-membership edges.
5. Create visual nodes from `visual.json`.
6. Add `TEMPORAL_NEXT` edges between consecutive visual nodes.
7. Add `ALIGNED_TO` edges from transcript nodes to temporally overlapping visual
   nodes.
8. Extract and canonicalize entities from transcript text, visual descriptions,
   state-change descriptions, detected visual entities, and OCR text.
9. Add entity nodes and `CONTAINS_ENTITY` edges from mentioning nodes to entity
   nodes.

After the graph is saved, `build_video_graph()` also exports GraphML and, by
default, pre-computes node embeddings for retrieval.

## Node Types

### TranscriptNode

Created from transcript sentence segments or raw transcript segments.

Important fields:

- `id`: `t_0000`, `t_0001`, ...
- `text`: transcript text
- `start`, `end`: timestamps in seconds
- `topic_id`: assigned topic id when discourse analysis creates one
- `schema`: topic discourse schema when available
- `sentence_id`: original transcript segment id

### VisualNode

Created from each entry in `visual.json["analyses"]`.

Important fields:

- `id`: `v_<clip_id>`
- `clip_id`: source clip identifier
- `start`, `end`: clip timestamps in seconds
- `visual_description`: visual caption for the clip
- `state_change_from_previous`: optional change description relative to the
  previous clip
- `detected_entities`: visual entities returned by the vision model
- `scene_type`: coarse scene label
- `ocr_text`: on-screen text when OCR is available
- `keyframes`: analyzed keyframe ids or paths

### TopicNode

Created by LLM discourse analysis when topic construction is enabled.

Important fields:

- `id`: topic identifier from the model response, usually `topic_0`, `topic_1`,
  ...
- `title`: short topic title
- `description`: topic description
- `schema`: one of `narrative`, `descriptive`, `informative`,
  `instructional`, or `argumentative`
- `keywords`: topic keywords
- `member_nodes`: transcript node ids assigned to the topic
- `start`, `end`: time span covered by the member transcript nodes

### EntityNode

Created by LLM entity extraction when entity construction is enabled.

Important fields:

- `id`: generated id such as `e_ab12cd34`
- `name`: canonical entity name
- `entity_type`: `person`, `organization`, `product`, `concept`, `location`,
  `event`, or `other`
- `aliases`: alternate names returned by the model
- `mentions`: transcript or visual node ids that contain the entity
- `start`, `end`: entities span the whole video by default

## Edge Types

### TEMPORAL_NEXT

Connects consecutive nodes within the same modality.

The builder creates:

- transcript-to-transcript temporal edges
- visual-to-visual temporal edges

These edges preserve timeline order and support temporal expansion during
retrieval.

### ALIGNED_TO

Connects transcript nodes to visual nodes when their timestamps overlap within
the configured temporal window.

Stored edge direction:

```text
TranscriptNode -> VisualNode
```

The edge weight is based on timestamp overlap:

```text
overlap_duration / transcript_duration
```

The graph stores this as a directed edge, but retrieval and subgraph expansion
can still treat graph neighbors as connected evidence.

### DISCOURSE_RELATION

Connects transcript nodes when the discourse-analysis model identifies a clear
semantic relation.

Supported relation labels are defined in `DiscourseRelation`, including:

```text
SUPPORT, COUNTER, EXPLAINS, DEFINES, EXAMPLE_OF, ELABORATES, SUMMARIZES,
CONTRASTS, STEP_BEFORE, STEP_AFTER, CAUSES, RESULTS_FROM, ENABLES,
INTRODUCES, CONCLUDES, TRANSITIONS, REFERENCES, QUOTES, CITES
```

The prompt asks the model to be conservative and include only clear relations.

### BELONGS_TO_TOPIC

Connects transcript nodes to topic nodes.

Stored edge direction:

```text
TranscriptNode -> TopicNode
```

### CONTAINS_ENTITY

Connects transcript or visual nodes to canonical entity nodes.

Stored edge direction:

```text
TranscriptNode -> EntityNode
VisualNode -> EntityNode
```

Visual entity mentions can come from visual descriptions, state-change
descriptions, detected entities, or OCR text.

## Configuration

The graph builder reads these config values through `build_video_graph()`:

```yaml
openai:
  text_model: "gpt-4o"
  embedding_model: "text-embedding-3-small"

graph:
  temperature: 0.3
  temporal_window: 5
  build_topic_nodes: true
  build_entity_nodes: true

processing:
  max_parallel_embeddings: 10
```

`graph.text_model` and `graph.embedding_model` can be provided to override the
corresponding `openai` values. If those graph-specific keys are not present, the
builder uses `openai.text_model` and `openai.embedding_model`.

## Outputs

`build_video_graph()` writes:

```text
graph.json
graph.graphml
embeddings.json
```

`graph.json` contains serialized nodes, edges, metadata, and summary statistics.
`graph.graphml` is a visualization-friendly export. `embeddings.json` stores
pre-computed embeddings for graph nodes so question answering can retrieve
evidence without embedding every node at query time.

## Retrieval Relevance

The graph is designed for reusable retrieval. Once a video has been processed,
question answering can:

- retrieve semantically relevant nodes using `embeddings.json`
- expand through temporal edges
- expand through transcript-visual alignment edges
- expand through topic and entity edges
- use discourse edges for semantically related transcript evidence

This is the main distinction between VideoGraph and flat retrieval over isolated
chunks: the graph stores reusable temporal, semantic, and multimodal structure
that can be traversed after initial retrieval.
