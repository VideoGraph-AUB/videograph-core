"""
Reinforced graph learning: post-build self-improvement loop.

The graph critiques its own captions for missing OBSERVABLE evidence (reactions,
vague interactions, undescribed background subjects, unexplained transitions),
re-perceives ONLY the probed clips (dense frames from the stored clip files), and
writes verified findings back into the clip nodes before rebuilding the graph.

Guards: probes are generated from the graph's own gaps (never from task questions);
findings must be grounded in pixels or the model returns NOT_VISIBLE and the probe
is rejected; cost is bounded (<= max_probes per video, one clip per probe).

Validated on NExT-QA failure-skewed videos: +2 (stills) / +3 (dense) per ~50 Q, 0 losses.
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

CRITIQUE_PROMPT = """You are auditing a video's knowledge graph for MISSING OBSERVABLE evidence.
Below are the per-clip captions in time order. Captions often under-describe:
- REACTIONS: how a person/animal visibly reacts when something happens (expression, body language)
- INTERACTIONS: vague phrases like "playfully interacting" that hide WHO did WHAT to WHOM
- SECONDARY SUBJECTS: background people/animals whose actions are never described
- TRANSITIONS: an entity's state changes between clips with no stated cause

List up to {max_probes} probes for the WEAKEST spots. Each probe must be answerable by LOOKING at that clip's frames.
Return JSON: {{"probes":[{{"clip_id":"clip_0002","probe":"specific visual question about observable actions/reactions","gap":"reaction|interaction|secondary|transition"}}]}}

Clips:
"""

REPERCEIVE_PROMPT = """Answer the question using ONLY what is visible in these frames sampled densely across one video clip (in time order).
Be specific about who does what (use visual identifiers). State only CONCRETE PHYSICAL ACTIONS you can
see — never interpretations of mood, intent, or feeling, and never hedged language ('appears to',
'seems', 'suggesting'). If not visible, reply exactly: NOT_VISIBLE
Answer in 1-2 sentences.

Question: """

# Speculative-register markers: interpretive findings flip answers the WRONG way as often as
# right (measured); concrete physical actions drive the wins. Reject hedged findings outright.
_SPECULATIVE = ("appears to", "appear to", "seems", "seem to", "suggesting", "suggests",
                "possibly", "likely", "might be", "may be")


def _dense_frames_b64(clip_path: Path, n_frames: int) -> list:
    try:
        import cv2
    except Exception:
        return []
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return []
    out = []
    try:
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        dur = total / max(fps, 1.0)
        for i in range(n_frames):
            cap.set(cv2.CAP_PROP_POS_MSEC, dur * (i + 0.5) / n_frames * 1000.0)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            h, w = fr.shape[:2]
            scale = 448.0 / max(w, 1)
            fr = cv2.resize(fr, (448, max(1, int(h * scale))))
            ok2, buf = cv2.imencode(".jpg", fr)
            if ok2:
                out.append(base64.b64encode(buf.tobytes()).decode())
    finally:
        cap.release()
    return out


def reinforce_video_graph(
    video_dir: str,
    text_model: str = "gpt-4o",
    vision_model: str = "gpt-4o",
    max_probes: int = 5,
    frames_per_probe: int = 8,
    api_key: Optional[str] = None,
    rebuild: bool = True,
) -> int:
    """
    Run one critique -> re-perceive -> reinforce pass over a processed video dir.
    Updates visual.json in place and rebuilds graph.json/embeddings. Returns the
    number of reinforced observations applied.
    """
    from .builder import build_video_graph

    video_dir = Path(video_dir)
    vis_path = video_dir / "visual.json"
    if not vis_path.exists():
        logger.warning(f"reinforce: no visual.json in {video_dir}")
        return 0

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    vis = json.loads(vis_path.read_text(encoding="utf-8"))
    all_rows = sorted(vis.get("analyses", []), key=lambda a: a.get("start", 0))
    # Exclude synthetic rows (e.g. vid_summary): they have no clip file to re-perceive.
    rows = [a for a in all_rows if (video_dir / "clips" / f"{a['clip_id']}.mp4").exists()]
    if not rows:
        return 0
    def _critique(window_rows, budget):
        timeline = "\n".join(
            f"[{a['clip_id']} {a.get('start', 0):.0f}-{a.get('end', 0):.0f}s] {(a.get('visual_description') or '')[:300]}"
            for a in window_rows
        )
        resp = client.chat.completions.create(
            model=text_model, temperature=0, seed=0, max_tokens=700,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": CRITIQUE_PROMPT.format(max_probes=budget) + timeline}],
        )
        try:
            return (json.loads(resp.choices[0].message.content).get("probes") or [])[:budget]
        except Exception:
            return []

    # Windowed critique: never pass more than WINDOW clip captions per call. Short videos
    # = one call (the validated path); long videos = one bounded critique per window, with
    # the probe budget scaled by length (capped) so cost stays linear-with-small-constant
    # rather than one unbounded mega-prompt.
    WINDOW = 30
    if len(rows) <= WINDOW:
        probes = _critique(rows, max_probes)
    else:
        n_windows = (len(rows) + WINDOW - 1) // WINDOW
        total_budget = min(max_probes * 3, n_windows * 2)  # long videos earn more probes, capped
        per_window = max(1, total_budget // n_windows)
        probes = []
        for i in range(0, len(rows), WINDOW):
            probes.extend(_critique(rows[i:i + WINDOW], per_window))
        probes = probes[:total_budget]

    by_id = {a["clip_id"]: a for a in rows}
    applied = 0
    for pr in probes:
        cid = pr.get("clip_id")
        q = pr.get("probe", "")
        a = by_id.get(cid)
        clip_path = video_dir / "clips" / f"{cid}.mp4"
        if not a or not q or not clip_path.exists():
            continue
        frames = _dense_frames_b64(clip_path, frames_per_probe)
        if not frames:
            continue
        content = [{"type": "text", "text": REPERCEIVE_PROMPT + q}]
        for b in frames:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}", "detail": "low"}})
        rr = client.chat.completions.create(
            model=vision_model, temperature=0, seed=0, max_tokens=120,
            messages=[{"role": "user", "content": content}],
        )
        finding = (rr.choices[0].message.content or "").strip()
        if not finding or "NOT_VISIBLE" in finding.upper():
            logger.info(f"reinforce {cid} [{pr.get('gap')}]: rejected (not visible)")
            continue
        if any(m in finding.lower() for m in _SPECULATIVE):
            logger.info(f"reinforce {cid} [{pr.get('gap')}]: rejected (speculative)")
            continue
        # Redundancy gate: only inject findings that ADD information the caption lacks.
        # Indiscriminate injection at scale perturbs retrieval and flips answers both ways
        # (measured: 52 ungated obs -> 6W/5L churn); novel-only injection keeps the wins.
        gate = client.chat.completions.create(
            model=text_model, temperature=0, seed=0, max_tokens=4,
            messages=[{"role": "user", "content":
                f"Caption: {a.get('visual_description','')}\nNew observation: {finding}\n"
                "Does the observation add specific NEW observable information not already "
                "stated or clearly implied by the caption? Answer YES or NO only."}],
        )
        if "YES" not in (gate.choices[0].message.content or "").upper():
            logger.info(f"reinforce {cid} [{pr.get('gap')}]: rejected (redundant)")
            continue
        a["visual_description"] = a.get("visual_description", "") + " Reinforced observation: " + finding
        applied += 1
        logger.info(f"reinforce {cid} [{pr.get('gap')}]: applied")

    if applied:
        # all_rows shares the mutated row dicts and keeps synthetic rows (vid_summary)
        vis["analyses"] = all_rows
        vis_path.write_text(json.dumps(vis, indent=2, ensure_ascii=False), encoding="utf-8")
        if rebuild:
            build_video_graph(str(video_dir))
        logger.info(f"reinforce: {applied} observations applied"
                    + (", graph rebuilt" if rebuild else " (rebuild deferred to pipeline)"))
    return applied
