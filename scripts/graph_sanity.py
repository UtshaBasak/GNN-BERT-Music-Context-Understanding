#!/usr/bin/env python
"""the gate you cannot skip: do similarity edges connect repeated sections?

    python scripts/graph_sanity.py [--dataset mtat] [--n 5] [--strict]

If k-NN similarity edges only ever link a segment to its immediate neighbours,
the graph carries nothing the temporal chain did not already carry, and Tasks 2
and 3 will both underperform for a reason no amount of GNN tuning will fix. That
is the failure this gate exists to catch, and it is worth catching *before*
spending GPU hours rather than after.

The check is quantitative, not eyeballed, because "the picture looks connected"
is not a decision procedure:

* **long-range fraction** — of the non-temporal, non-self edges, what share span
  ``|i - j| >= min_lag`` (default 3)? A track with verse/chorus/verse structure
  should link its repeats across many segments. Near zero means the features are
  too smooth to distinguish a chorus from the bar before it.
* **repeat recall** — of the genuinely most-similar distant segment pairs (top
  decile of cosine at lag >= min_lag), how many did the k-NN graph actually
  connect? This is deliberately *not* circular: k-NN optimises global similarity,
  so it can and does spend all k edges on near neighbours, missing the repeats.
* **temporal-only degree** — how many nodes have no similarity edge at all beyond
  their chain neighbours.

Tracks are chosen by measured repetition (highest distant self-similarity), not
at random, because the gate asks about tracks that *have* repeated structure.

Renders and a JSON verdict go to ``results/plots/graph_sanity/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.audio_features import NODE_FEAT_DIM  # noqa: E402
from src.graph_builder import (  # noqa: E402
    build_segment_graph,
    cosine_similarity_matrix,
    visualise_graph,
)
from src.utils import (  # noqa: E402
    ensure_dir,
    get_logger,
    load_config,
    parse_overrides,
    resolve_path,
    save_json,
)

LOGGER = get_logger("gbmc.graph_sanity")

#: Below this share of long-range similarity edges the graph is effectively a
#: chain and the node features need to change (shorter window, higher k, or
#: richer features) before any GNN result is meaningful.
MIN_LONG_RANGE_FRACTION = 0.30
MIN_REPEAT_RECALL = 0.30


def repetition_score(feats: np.ndarray, min_lag: int = 3) -> float:
    """How strongly a track repeats itself at a distance.

    Mean of the top decile of off-diagonal cosine similarities at lag >= min_lag.
    High means verse/chorus/verse; low means through-composed.
    """
    n = feats.shape[0]
    if n <= min_lag + 1:
        return float("nan")
    sim = cosine_similarity_matrix(feats)
    lag = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    distant = sim[lag >= min_lag]
    if distant.size == 0:
        return float("nan")
    return float(np.mean(np.sort(distant)[-max(1, distant.size // 10):]))


def analyse_graph(data, feats: np.ndarray, min_lag: int = 3) -> dict:
    """Measure whether the similarity edges reach beyond the temporal chain."""
    n = int(data.num_nodes)
    ei = data.edge_index.cpu().numpy()
    ea = data.edge_attr.cpu().numpy()

    non_self = ei[0] != ei[1]
    is_temporal = ea[:, 0] > 0.5
    sim_mask = non_self & ~is_temporal
    src, dst = ei[0][sim_mask], ei[1][sim_mask]
    lags = np.abs(src - dst)

    long_range = int(np.sum(lags >= min_lag))
    n_sim = int(sim_mask.sum())

    # the pairs a human would call "repeats": most similar at a distance
    sim_matrix = cosine_similarity_matrix(feats)
    lag_matrix = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    candidates = [(i, j, sim_matrix[i, j])
                  for i in range(n) for j in range(i + 1, n)
                  if lag_matrix[i, j] >= min_lag]
    repeat_recall = float("nan")
    n_repeat_pairs = 0
    if candidates:
        candidates.sort(key=lambda t: -t[2])
        top = candidates[: max(1, len(candidates) // 10)]
        n_repeat_pairs = len(top)
        connected = {(int(a), int(b)) for a, b in zip(src, dst)}
        hit = sum(1 for i, j, _ in top
                  if (i, j) in connected or (j, i) in connected)
        repeat_recall = hit / len(top)

    sim_degree = np.bincount(src, minlength=n) if n_sim else np.zeros(n, dtype=int)
    return {
        "track_id": getattr(data, "track_id", ""),
        "dataset": getattr(data, "dataset", ""),
        "num_nodes": n,
        "num_edges": int(ei.shape[1]),
        "n_similarity_edges": n_sim,
        "long_range_edges": long_range,
        "long_range_fraction": float(long_range / n_sim) if n_sim else 0.0,
        "mean_similarity_lag": float(np.mean(lags)) if lags.size else 0.0,
        "max_similarity_lag": int(np.max(lags)) if lags.size else 0,
        "repeat_recall": repeat_recall,
        "n_repeat_pairs_considered": n_repeat_pairs,
        "nodes_without_similarity_edges": int(np.sum(sim_degree == 0)),
        "repetition_score": repetition_score(feats, min_lag),
    }


def load_candidates(cfg, dataset: str, pool: int) -> list[tuple[str, np.ndarray]]:
    """Read a pool of real tracks from the feature cache."""
    import h5py

    processed = resolve_path(cfg["paths"]["processed"])
    h5_path = processed / f"features_{dataset}.h5"
    if not h5_path.exists():
        h5_path = processed / "features.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"no feature cache for {dataset} at {h5_path}. Run "
            f"`python -m src.audio_features --datasets {dataset}` first."
        )

    stats_path = processed / f"norm_stats_{dataset}.json"
    norm = None
    if stats_path.exists():
        norm = json.loads(stats_path.read_text(encoding="utf-8"))
        if norm.get("split") != "train":
            raise RuntimeError(f"{stats_path} is not train-split provenance")

    out = []
    with h5py.File(h5_path, "r") as store:
        for key in list(store.keys())[: int(pool)]:
            feats = np.asarray(store[key][...], dtype=np.float32)
            if feats.shape[1] != NODE_FEAT_DIM:
                continue
            if norm is not None:
                from src.audio_features import apply_norm

                feats = apply_norm(feats, norm)
            out.append((str(key), feats))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="A4.4 graph sanity gate.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default="mtat")
    parser.add_argument("--n", type=int, default=5, help="tracks to render")
    parser.add_argument("--pool", type=int, default=400,
                        help="tracks to search for repeated structure")
    parser.add_argument("--min-lag", type=int, default=3)
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if the gate fails")
    parser.add_argument("--out-dir", default="results/plots/graph_sanity")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args(argv)

    cfg = load_config(args.config, parse_overrides(args.override))
    out_dir = ensure_dir(args.out_dir)

    candidates = load_candidates(cfg, args.dataset, args.pool)
    if not candidates:
        LOGGER.error("no usable tracks in the cache")
        return 1

    # pick the most repetitive tracks: the gate is about verse/chorus structure
    scored = [(key, feats, repetition_score(feats, args.min_lag))
              for key, feats in candidates]
    scored = [row for row in scored if np.isfinite(row[2])]
    scored.sort(key=lambda row: -row[2])
    chosen = scored[: int(args.n)]
    LOGGER.info("selected %d of %d tracks by repetition score (%.3f to %.3f)",
                len(chosen), len(scored), chosen[0][2], chosen[-1][2])

    reports = []
    for i, (key, feats, score) in enumerate(chosen):
        data = build_segment_graph(feats, cfg, track_id=key, dataset=args.dataset)
        report = analyse_graph(data, feats, args.min_lag)
        report["render"] = str(visualise_graph(
            data, out_path=out_dir / f"sanity_{i:02d}_{key}.png",
            title=f"{key}  repetition={score:.3f}  "
                  f"long-range={report['long_range_fraction']:.2f}  "
                  f"repeat-recall={report['repeat_recall']:.2f}",
        ))
        reports.append(report)
        LOGGER.info(
            "%-22s nodes=%-3d sim_edges=%-4d long_range=%.2f repeat_recall=%.2f "
            "mean_lag=%.1f isolated=%d",
            key, report["num_nodes"], report["n_similarity_edges"],
            report["long_range_fraction"], report["repeat_recall"],
            report["mean_similarity_lag"], report["nodes_without_similarity_edges"],
        )

    frame = pd.DataFrame(reports)
    verdict = {
        "dataset": args.dataset,
        "n_tracks": len(reports),
        "min_lag": args.min_lag,
        "mean_long_range_fraction": float(frame["long_range_fraction"].mean()),
        "mean_repeat_recall": float(frame["repeat_recall"].mean()),
        "mean_similarity_lag": float(frame["mean_similarity_lag"].mean()),
        "tracks_with_isolated_nodes": int((frame["nodes_without_similarity_edges"] > 0).sum()),
        "thresholds": {"long_range_fraction": MIN_LONG_RANGE_FRACTION,
                       "repeat_recall": MIN_REPEAT_RECALL},
        "config": {"window_s": cfg["segmentation"]["window_s"],
                   "overlap": cfg["segmentation"]["overlap"],
                   "knn_k": cfg["graph"]["knn_k"]},
        "per_track": reports,
    }
    verdict["passed"] = bool(
        verdict["mean_long_range_fraction"] >= MIN_LONG_RANGE_FRACTION
        and verdict["mean_repeat_recall"] >= MIN_REPEAT_RECALL
    )
    save_json(verdict, out_dir / "graph_sanity.json")

    print(f"\n{'=' * 72}\nA4.4 GRAPH SANITY GATE — {args.dataset}\n{'=' * 72}")
    print(f"  window_s={verdict['config']['window_s']}  "
          f"overlap={verdict['config']['overlap']}  knn_k={verdict['config']['knn_k']}")
    print(f"  mean long-range fraction : {verdict['mean_long_range_fraction']:.3f}  "
          f"(need >= {MIN_LONG_RANGE_FRACTION})")
    print(f"  mean repeat recall       : {verdict['mean_repeat_recall']:.3f}  "
          f"(need >= {MIN_REPEAT_RECALL})")
    print(f"  mean similarity lag      : {verdict['mean_similarity_lag']:.2f} segments")
    print(f"  tracks with isolated nodes: {verdict['tracks_with_isolated_nodes']}")
    print(f"\n  VERDICT: {'PASS' if verdict['passed'] else 'FAIL'}")
    if not verdict["passed"]:
        print("\n  Similarity edges are not reaching repeated sections. Try, in order:")
        print("    1. shorter windows   --override segmentation.window_s=2.0")
        print("    2. higher k          --override graph.knn_k=6")
        print("    3. add chroma std to the node feature (changes the 96-dim contract)")
        print("    4. beat-synchronous  --override segmentation.mode=beat_sync")
    print(f"\n  renders + verdict: {out_dir}\n")

    return 0 if (verdict["passed"] or not args.strict) else 2


if __name__ == "__main__":
    raise SystemExit(main())
