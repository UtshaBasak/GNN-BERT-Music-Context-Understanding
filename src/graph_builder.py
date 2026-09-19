"""Turn per-segment audio features into the graphs the GNNs consume.

Design decisions worth defending in the report:

* **Similarity edges are k-NN, not a cosine threshold.** A single global tau
  gives isolated nodes on a sparse, through-composed track and a near-clique on
  a loop-based one -- so the graph topology ends up encoding "how repetitive is
  this track" rather than "which segments belong together", and the GNN's
  receptive field varies wildly across the batch. Fixed k keeps message passing
  comparable across tracks. The cosine value is still kept in ``edge_attr`` so
  the model can learn to discount weak neighbours.
* **Edges are stored bidirectionally.** ``SAGEConv``/``GATv2Conv`` propagate
  along the given direction only; a one-way temporal chain would let information
  flow forward in time but never back.
* **Self-loops are explicit**, so a node's own features survive a mean
  aggregation on a graph where it has few informative neighbours.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData

from .utils import REAL, ensure_dir, get_logger, resolve_path

LOGGER = get_logger("gbmc.graph")

__all__ = [
    "build_segment_graph",
    "build_chord_graph",
    "build_hetero_graph",
    "rewire_edges",
    "visualise_graph",
    "export_sample_graphs",
    "knn_edge_index",
    "cosine_similarity_matrix",
]

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# edge construction primitives
# --------------------------------------------------------------------------- #
def cosine_similarity_matrix(feats: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity, safe for all-zero rows."""
    feats = np.asarray(feats, dtype=np.float64)
    norms = np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), _EPS)
    unit = feats / norms
    return np.clip(unit @ unit.T, -1.0, 1.0)


def knn_edge_index(feats: np.ndarray, k: int) -> np.ndarray:
    """Directed k-NN edges ``i -> j`` in cosine space, excluding self.

    Every node emits exactly ``min(k, n - 1)`` edges, which is the property that
    makes the receptive field constant across tracks. Returned as ``[2, n*k]``.
    """
    feats = np.asarray(feats, dtype=np.float64)
    n = feats.shape[0]
    k = int(min(max(k, 0), max(n - 1, 0)))
    if n < 2 or k == 0:
        return np.zeros((2, 0), dtype=np.int64)

    sim = cosine_similarity_matrix(feats)
    np.fill_diagonal(sim, -np.inf)
    # mergesort keeps the order stable so identical segments tie-break by index.
    neighbours = np.argsort(-sim, axis=1, kind="mergesort")[:, :k]
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = neighbours.reshape(-1).astype(np.int64)
    return np.stack([src, dst], axis=0)


def _temporal_edge_index(n: int) -> np.ndarray:
    """Bidirectional chain ``i <-> i+1``."""
    if n < 2:
        return np.zeros((2, 0), dtype=np.int64)
    fwd = np.arange(n - 1, dtype=np.int64)
    src = np.concatenate([fwd, fwd + 1])
    dst = np.concatenate([fwd + 1, fwd])
    return np.stack([src, dst], axis=0)


def _symmetrise(edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return edges
    return np.concatenate([edges, edges[::-1, :]], axis=1)


def _assemble_edges(
    n: int,
    temporal: np.ndarray,
    similarity: np.ndarray,
    sim_matrix: np.ndarray,
    add_self_loops: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge edge sources into a deduplicated COO set with ``[is_temporal, cos]``.

    A pair that is both consecutive in time and a mutual nearest neighbour keeps
    ``is_temporal = 1``; the cosine channel is filled in for every edge either
    way, so the two channels stay independently interpretable.
    """
    temporal_pairs = {(int(a), int(b)) for a, b in zip(*temporal)} if temporal.size else set()
    all_pairs: dict[tuple[int, int], float] = {}

    for pair in temporal_pairs:
        all_pairs[pair] = 1.0
    if similarity.size:
        for a, b in zip(*similarity):
            all_pairs.setdefault((int(a), int(b)), 0.0)
    if add_self_loops:
        for i in range(n):
            all_pairs.setdefault((i, i), 0.0)

    if not all_pairs:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, 2), dtype=torch.float32),
        )

    ordered = sorted(all_pairs.items())
    src = np.array([p[0][0] for p in ordered], dtype=np.int64)
    dst = np.array([p[0][1] for p in ordered], dtype=np.int64)
    is_temporal = np.array([p[1] for p in ordered], dtype=np.float32)
    cos = sim_matrix[src, dst].astype(np.float32)
    edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
    edge_attr = torch.from_numpy(np.stack([is_temporal, cos], axis=1)).float()
    return edge_index, edge_attr


# --------------------------------------------------------------------------- #
# the frozen data contract
# --------------------------------------------------------------------------- #
def _tag_tensor(y_tags, n_tags_hint: int | None = None) -> torch.Tensor:
    """Multi-hot tags as ``[1, n_tags]``; ``-1`` everywhere if unavailable.

    Stored with a leading batch dim so PyG's default collation stacks tracks
    into ``[B, n_tags]`` instead of concatenating into one long vector.
    """
    if y_tags is None:
        if not n_tags_hint:
            return torch.full((1, 1), -1.0, dtype=torch.float32)
        return torch.full((1, int(n_tags_hint)), -1.0, dtype=torch.float32)
    arr = np.asarray(y_tags, dtype=np.float32).reshape(1, -1)
    return torch.from_numpy(arr).float()


def _scalar(value, dtype, missing):
    if value is None:
        value = missing
    try:
        value = missing if value is None else value
        if dtype == torch.float32 and value is not None and not np.isfinite(float(value)):
            value = float("nan")
    except (TypeError, ValueError):
        value = missing
    return torch.tensor([value], dtype=dtype)


def _attach_labels(data: Data, labels: dict, n_tags_hint: int | None = None) -> Data:
    """Populate the contract fields, using the sentinel rule for absences."""
    data.y_tags = _tag_tensor(labels.get("y_tags"), n_tags_hint)
    data.y_genre = _scalar(labels.get("y_genre"), torch.long, -1)
    data.y_valence = _scalar(labels.get("y_valence"), torch.float32, float("nan"))
    data.y_arousal = _scalar(labels.get("y_arousal"), torch.float32, float("nan"))
    data.track_id = str(labels.get("track_id", ""))
    data.artist_id = str(labels.get("artist_id", ""))
    data.dataset = str(labels.get("dataset", ""))
    data.split = str(labels.get("split", ""))
    data.text = str(labels.get("text", ""))
    # Additive contract field. `dataset` names which corpus the label pattern
    # imitates, so it cannot distinguish fake from real -- the synthetic
    # generator reuses those names on purpose. See utils.detect_provenance.
    data.provenance = str(labels.get("provenance", REAL))
    for extra in ("audio_path", "duration_s"):
        if extra in labels:
            setattr(data, extra, labels[extra])
    return data


def build_segment_graph(seg_feats, cfg, **labels) -> Data:
    """Build the segment-level ``Data`` object from ``[num_seg, 96]`` features.

    ``labels`` accepts every field of the frozen contract; anything omitted gets
    its sentinel (``-1`` for tags/genre, ``nan`` for valence/arousal).
    """
    feats = np.asarray(seg_feats, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError(f"seg_feats must be 2-D [num_seg, F], got {feats.shape}")

    graph_cfg = cfg["graph"]
    expected_dim = int(graph_cfg["node_feat_dim"])
    if feats.shape[1] != expected_dim:
        raise AssertionError(
            f"node feature dim is {feats.shape[1]}, contract requires {expected_dim}. "
            "Check the canonical feature order in audio_features.segment_features."
        )

    seg_cfg = cfg.get("segmentation", {})
    max_nodes = int(seg_cfg.get("max_nodes", feats.shape[0]) or feats.shape[0])
    if feats.shape[0] > max_nodes:
        # Uniformly subsample rather than truncate: a 29 s clip should not be
        # represented only by its first 10 seconds.
        keep = np.linspace(0, feats.shape[0] - 1, max_nodes).round().astype(int)
        feats = feats[keep]
    n = feats.shape[0]
    if n == 0:
        raise ValueError("cannot build a graph from zero segments")

    sim_matrix = cosine_similarity_matrix(feats)
    temporal = (
        _temporal_edge_index(n)
        if graph_cfg.get("temporal_edges", True)
        else np.zeros((2, 0), dtype=np.int64)
    )
    similarity = (
        _symmetrise(knn_edge_index(feats, int(graph_cfg["knn_k"])))
        if graph_cfg.get("similarity_edges", True)
        else np.zeros((2, 0), dtype=np.int64)
    )
    edge_index, edge_attr = _assemble_edges(
        n, temporal, similarity, sim_matrix, bool(graph_cfg.get("add_self_loops", True))
    )

    data = Data(
        x=torch.from_numpy(feats).float(),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    data.num_nodes = n
    _attach_labels(data, labels, labels.get("n_tags"))
    assert data.x.shape[1] == expected_dim
    return data


def build_chord_graph(chord_seq: Sequence[str], chroma=None, cfg=None, **labels) -> Data:
    """Chord-transition graph over the chords actually observed in the track.

    Nodes are the distinct chord symbols present (at most the 25-symbol
    vocabulary), each carrying its vocabulary index plus the mean chroma of the
    frames where it was active. Edges are observed transitions, weighted by the
    row-normalised transition probability -- i.e. "given we are on C, how often
    do we go to G", which is the quantity a harmony prior is actually about.
    """
    from .chords import CHORD_VOCAB, chord_transition_matrix

    cfg = cfg or {}
    graph_cfg = cfg.get("graph", {}) if isinstance(cfg, dict) else {}
    feat_dim = int(graph_cfg.get("chord_feat_dim", 13))
    vocab = list(CHORD_VOCAB)
    vocab_index = {name: i for i, name in enumerate(vocab)}

    seq = [str(c) for c in chord_seq] if len(chord_seq) else ["N"]
    present = sorted({c for c in seq if c in vocab_index}, key=lambda c: vocab_index[c])
    if not present:
        present = ["N"]
    local_index = {name: i for i, name in enumerate(present)}

    chroma_arr = None
    if chroma is not None:
        chroma_arr = np.asarray(chroma, dtype=np.float32)
        if chroma_arr.ndim == 2 and chroma_arr.shape[0] != 12 and chroma_arr.shape[1] == 12:
            chroma_arr = chroma_arr.T

    x = np.zeros((len(present), feat_dim), dtype=np.float32)
    chord_ids = np.zeros(len(present), dtype=np.int64)
    for name, i in local_index.items():
        chord_ids[i] = vocab_index[name]
        x[i, 0] = vocab_index[name] / max(len(vocab) - 1, 1)
        if chroma_arr is not None and chroma_arr.shape[0] == 12:
            frames = [t for t, c in enumerate(seq) if c == name and t < chroma_arr.shape[1]]
            if frames:
                pooled = chroma_arr[:, frames].mean(axis=1)
                width = min(feat_dim - 1, pooled.size)
                x[i, 1 : 1 + width] = pooled[:width]

    transitions = chord_transition_matrix(seq, vocab)
    src_list, dst_list, weights = [], [], []
    for a in present:
        row = transitions[vocab_index[a]]
        total = row.sum()
        if total <= 0:
            continue
        for b in present:
            w = row[vocab_index[b]]
            if w > 0:
                src_list.append(local_index[a])
                dst_list.append(local_index[b])
                weights.append(float(w / total))

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    else:  # single-chord track: keep one self-loop so message passing is defined
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.ones((1, 1), dtype=torch.float32)

    data = Data(x=torch.from_numpy(x), edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = len(present)
    data.chord_ids = torch.from_numpy(chord_ids)
    data.chord_names = present
    _attach_labels(data, labels, labels.get("n_tags"))
    return data


def build_hetero_graph(seg_data: Data, chord_data: Data, seg2chord) -> HeteroData:
    """Assemble the segment/chord heterogeneous graph.

    ``seg2chord`` maps segment node index -> chord node index (local to
    ``chord_data``); it may be a list, an array, or a list of lists when a
    segment spans several chords.
    """
    het = HeteroData()
    het["segment"].x = seg_data.x.clone()
    het["segment"].num_nodes = int(seg_data.num_nodes)
    het["chord"].x = chord_data.x.clone()
    het["chord"].num_nodes = int(chord_data.num_nodes)
    if hasattr(chord_data, "chord_ids"):
        het["chord"].chord_ids = chord_data.chord_ids.clone()

    # split the homogeneous segment edges back into their two semantic types
    ei = seg_data.edge_index
    ea = seg_data.edge_attr
    if ea is not None and ea.numel():
        temporal_mask = ea[:, 0] > 0.5
    else:  # pragma: no cover - only when edge_attr was dropped
        temporal_mask = torch.zeros(ei.shape[1], dtype=torch.bool)
    sim_mask = ~temporal_mask & (ei[0] != ei[1])

    het["segment", "next", "segment"].edge_index = ei[:, temporal_mask]
    het["segment", "next", "segment"].edge_attr = ea[temporal_mask]
    het["segment", "similar", "segment"].edge_index = ei[:, sim_mask]
    het["segment", "similar", "segment"].edge_attr = ea[sim_mask]

    seg_src, chord_dst = [], []
    for seg_idx, chord_idx in enumerate(seg2chord if seg2chord is not None else []):
        targets = chord_idx if isinstance(chord_idx, (list, tuple, np.ndarray)) else [chord_idx]
        for target in targets:
            target = int(target)
            if 0 <= target < het["chord"].num_nodes:
                seg_src.append(seg_idx)
                chord_dst.append(target)
    contains = torch.tensor([seg_src, chord_dst], dtype=torch.long) if seg_src else torch.zeros(
        (2, 0), dtype=torch.long
    )
    het["segment", "contains", "chord"].edge_index = contains
    het["chord", "in", "segment"].edge_index = contains.flip(0)

    het["chord", "transitions", "chord"].edge_index = chord_data.edge_index.clone()
    het["chord", "transitions", "chord"].edge_attr = chord_data.edge_attr.clone()

    for field in ("y_tags", "y_genre", "y_valence", "y_arousal",
                  "track_id", "artist_id", "dataset", "split", "text"):
        if hasattr(seg_data, field):
            setattr(het, field, getattr(seg_data, field))
    return het


# --------------------------------------------------------------------------- #
# controls and rendering
# --------------------------------------------------------------------------- #
def rewire_edges(data: Data, preserve_degree: bool = True, seed: int = 42) -> Data:
    """Random-rewiring control: same degrees, destroyed structure.

    This is the ablation that separates "the GNN uses musical structure" from
    "the GNN is a fancy pooled-feature MLP". If accuracy survives rewiring, the
    topology was never carrying signal. Degree-preserving mode uses double-edge
    swaps so degree distribution -- and therefore aggregation statistics -- is
    untouched.
    """
    rng = np.random.default_rng(seed)
    out = data.clone()
    ei = data.edge_index.cpu().numpy()
    n = int(data.num_nodes)
    if ei.size == 0 or n < 4:
        return out

    self_loops = [(int(a), int(b)) for a, b in zip(*ei) if a == b]
    undirected = sorted({(min(int(a), int(b)), max(int(a), int(b)))
                         for a, b in zip(*ei) if a != b})
    if len(undirected) < 2:
        return out

    edges = [list(e) for e in undirected]
    edge_set = {tuple(e) for e in edges}

    if preserve_degree:
        # Double-edge swap: (a,b),(c,d) -> (a,d),(c,b). Every endpoint keeps its
        # degree exactly; only who-connects-to-whom changes.
        for _ in range(10 * len(edges)):
            i, j = rng.integers(0, len(edges), size=2)
            if i == j:
                continue
            a, b = edges[i]
            c, d = edges[j]
            if rng.random() < 0.5:
                c, d = d, c
            if len({a, b, c, d}) < 4:
                continue
            new1 = (min(a, d), max(a, d))
            new2 = (min(c, b), max(c, b))
            if new1 in edge_set or new2 in edge_set:
                continue
            edge_set.discard((min(a, b), max(a, b)))
            edge_set.discard((min(c, d), max(c, d)))
            edge_set.add(new1)
            edge_set.add(new2)
            edges[i] = list(new1)
            edges[j] = list(new2)
    else:
        m = len(edges)
        edge_set = set()
        while len(edge_set) < m:
            a, b = rng.integers(0, n, size=2)
            if a == b:
                continue
            edge_set.add((int(min(a, b)), int(max(a, b))))

    pairs = sorted(edge_set)
    src = [a for a, b in pairs] + [b for a, b in pairs] + [a for a, _ in self_loops]
    dst = [b for a, b in pairs] + [a for a, b in pairs] + [b for _, b in self_loops]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    feats = data.x.detach().cpu().numpy()
    sim = cosine_similarity_matrix(feats)
    cos = sim[np.array(src, dtype=int), np.array(dst, dtype=int)].astype(np.float32)
    is_temporal = np.zeros_like(cos)  # rewired edges carry no temporal meaning
    out.edge_index = edge_index
    out.edge_attr = torch.from_numpy(np.stack([is_temporal, cos], axis=1)).float()
    out.num_nodes = n
    return out


def visualise_graph(data: Data, node_scores=None, out_path=None, title: str | None = None):
    """Render a segment graph with networkx; optionally shade nodes by score."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import networkx as nx

    n = int(data.num_nodes)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    ei = data.edge_index.cpu().numpy()
    ea = data.edge_attr.cpu().numpy() if data.edge_attr is not None else None

    temporal, similar = [], []
    for idx, (a, b) in enumerate(zip(*ei)):
        a, b = int(a), int(b)
        if a == b:
            continue
        graph.add_edge(a, b)
        if ea is not None and ea[idx, 0] > 0.5:
            temporal.append((a, b))
        else:
            similar.append((a, b))

    # circular layout keeps time legible: node k sits at angle 2*pi*k/n
    pos = {i: (np.cos(2 * np.pi * i / max(n, 1)), np.sin(2 * np.pi * i / max(n, 1)))
           for i in range(n)}

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx_edges(graph, pos, edgelist=similar, ax=ax, alpha=0.35,
                           edge_color="#7f8fa6", style="dashed")
    nx.draw_networkx_edges(graph, pos, edgelist=temporal, ax=ax, alpha=0.9,
                           edge_color="#2f3640", width=1.8)
    if node_scores is not None:
        scores = np.asarray(node_scores, dtype=float).reshape(-1)[:n]
        nodes = nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=scores,
                                       cmap="viridis", node_size=320)
        fig.colorbar(nodes, ax=ax, shrink=0.75, label="attention / score")
    else:
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="#40739e", node_size=320)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_color="white")

    label = title or f"{getattr(data, 'dataset', '')} / {getattr(data, 'track_id', '')}"
    ax.set_title(f"{label}\n{n} segments, {graph.number_of_edges()} undirected edges",
                 fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    if out_path:
        out = resolve_path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140)
        plt.close(fig)
        return out
    return fig


def _graph_summary(data: Data) -> dict[str, Any]:
    ei = data.edge_index
    n = int(data.num_nodes)
    non_self = ei[0] != ei[1]
    degrees = np.bincount(ei[0][non_self].cpu().numpy(), minlength=n) if n else np.array([])
    ea = data.edge_attr
    y_tags = data.y_tags.cpu().numpy().reshape(-1)
    return {
        "track_id": getattr(data, "track_id", ""),
        "artist_id": getattr(data, "artist_id", ""),
        "dataset": getattr(data, "dataset", ""),
        "provenance": getattr(data, "provenance", REAL),
        "split": getattr(data, "split", ""),
        "num_nodes": n,
        "num_edges": int(ei.shape[1]),
        "num_self_loops": int((~non_self).sum().item()),
        "node_feat_dim": int(data.x.shape[1]),
        "edge_attr_dim": int(ea.shape[1]) if ea is not None else 0,
        "mean_degree": float(degrees.mean()) if degrees.size else 0.0,
        "min_degree": int(degrees.min()) if degrees.size else 0,
        "temporal_edges": int((ea[:, 0] > 0.5).sum().item()) if ea is not None else 0,
        "mean_edge_cosine": float(ea[:, 1].mean().item()) if ea is not None and ea.numel() else 0.0,
        "n_positive_tags": int((y_tags > 0).sum()) if not np.all(y_tags < 0) else -1,
        "y_genre": int(data.y_genre.item()),
        "y_valence": float(data.y_valence.item()),
        "y_arousal": float(data.y_arousal.item()),
        "text": (getattr(data, "text", "") or "")[:240],
    }


def export_sample_graphs(manifest, out_dir="data/processed/sample_graphs", n: int = 20) -> list:
    """Write >= ``n`` graphs plus JSON summaries, as runnable examples.

    ``manifest`` may be a list of ``Data`` objects, a directory of ``.pt``
    graphs, or a DataFrame/manifest path that a
    :class:`~src.datasets.MusicGraphDataset` can be built from.
    """
    out = ensure_dir(out_dir)
    graphs = _coerce_graph_source(manifest, n)
    if not graphs:
        raise RuntimeError(
            "no graphs available to export -- run `python -m src.synthetic` or "
            "build a real manifest first"
        )

    written = []
    for idx, data in enumerate(graphs[:max(n, len(graphs[:n]))]):
        track = str(getattr(data, "track_id", f"graph{idx:03d}")).replace("/", "_")
        stem = f"{idx:03d}_{track}"[:80]
        pt_path = out / f"{stem}.pt"
        torch.save(data, pt_path)
        summary = _graph_summary(data)
        with open(out / f"{stem}.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        written.append(pt_path)

    index = {
        "n_graphs": len(written),
        "files": [p.name for p in written],
        "contract": {
            "x": "[num_nodes, 96] float32",
            "edge_index": "[2, num_edges] int64, COO, bidirectional",
            "edge_attr": "[num_edges, 2] float32 = [is_temporal, cosine_sim]",
            "y_tags": "[1, num_tags] float32, -1 = label absent for this dataset",
            "y_genre": "[1] int64, -1 = unavailable",
            "y_valence": "[1] float32, nan = unavailable",
            "y_arousal": "[1] float32, nan = unavailable",
            "provenance": "str, 'real' or 'synthetic'",
        },
        "provenance": sorted({getattr(g, "provenance", REAL) for g in graphs}),
    }
    with open(out / "index.json", "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    LOGGER.info("exported %d sample graphs to %s", len(written), out)
    return written


def _coerce_graph_source(manifest, n: int) -> list:
    """Accept the several shapes callers realistically have on hand."""
    import pandas as pd

    if isinstance(manifest, (list, tuple)):
        return list(manifest)[:n]

    if isinstance(manifest, (str, Path)):
        path = resolve_path(manifest)
        if path.is_dir():
            files = sorted(path.glob("*.pt"))[:n]
            return [torch.load(f, weights_only=False) for f in files]
        if path.suffix == ".csv":
            manifest = pd.read_csv(path)

    if isinstance(manifest, pd.DataFrame):
        from .datasets import MusicGraphDataset

        ds = MusicGraphDataset(manifest)
        return [ds[i] for i in range(min(n, len(ds)))]

    if hasattr(manifest, "__getitem__") and hasattr(manifest, "__len__"):
        return [manifest[i] for i in range(min(n, len(manifest)))]
    return []
