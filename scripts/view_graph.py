#!/usr/bin/env python
"""Render a stored segment graph as a picture.

    python scripts/view_graph.py                      # list the samples
    python scripts/view_graph.py 0                    # render sample 0
    python scripts/view_graph.py deam_10              # render by track id
    python scripts/view_graph.py --all                # render all 20 samples
    python scripts/view_graph.py path/to/graph.pt     # render any .pt

A `.pt` is a pickled PyTorch Geometric ``Data`` object, so an editor shows raw
bytes. Two readable views already exist beside it: the ``.json`` sibling written
by ``export_sample_graphs.py`` carries the summary statistics, and this script
draws the topology.

Nodes are laid out in a ring in temporal order, which makes the two edge types
legible against each other: the solid ring is the temporal chain between
consecutive segments, and the dashed chords are the k-nearest-neighbour
similarity edges. A track whose chords cluster into a few bundles is repetitive;
one whose chords are spread thin is through-composed. That contrast is the whole
reason the representation is a graph rather than a sequence.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dir, get_logger, project_root  # noqa: E402

LOGGER = get_logger("gbmc.viewgraph")

NODE = "#3b6ea5"
TEMPORAL = "#22303f"
SIMILAR = "#9fb6cd"


def _samples() -> list[Path]:
    return sorted((project_root() / "data" / "processed" / "sample_graphs").glob("*.pt"))


def resolve(token: str | None) -> Path | None:
    """Accept an index, a track id, or a path."""
    if token is None:
        return None
    candidate = Path(token)
    if candidate.exists():
        return candidate
    samples = _samples()
    if token.isdigit() and int(token) < len(samples):
        return samples[int(token)]
    for path in samples:
        if token in path.stem:
            return path
    return None


def render(path: Path, out_dir: Path):
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    data = torch.load(path, weights_only=False)
    n = int(data.num_nodes)
    edge_index = data.edge_index.numpy()
    edge_attr = data.edge_attr.numpy()

    # ring layout in temporal order
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]
    xs = [math.cos(a) for a in angles]
    ys = [-math.sin(a) for a in angles]

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.set_aspect("equal")
    ax.set_axis_off()

    seen = set()
    n_temporal = n_similar = 0
    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        if i == j or (j, i) in seen:
            continue
        seen.add((i, j))
        is_temporal = bool(edge_attr[k][0] > 0.5)
        cosine = float(edge_attr[k][1])
        if is_temporal:
            n_temporal += 1
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=TEMPORAL, lw=2.0, zorder=1)
        else:
            n_similar += 1
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=SIMILAR, lw=0.9,
                    ls="--", alpha=0.35 + 0.55 * max(0.0, min(1.0, cosine)),
                    zorder=0)

    ax.scatter(xs, ys, s=340, color=NODE, zorder=2, edgecolor="white", linewidth=1.4)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x, y, str(i), ha="center", va="center", color="white",
                fontsize=8, zorder=3)

    track = getattr(data, "track_id", path.stem)
    corpus = getattr(data, "dataset", "?")
    text = str(getattr(data, "text", "") or "")
    ax.set_title(
        f"{track}  ({corpus})\n"
        f"{n} segments · {n_temporal} temporal edges · {n_similar} similarity edges\n"
        f"{text[:70]}",
        fontsize=9.5, linespacing=1.5)

    ax.margins(0.12)
    fig.tight_layout()
    ensure_dir(out_dir)
    out = out_dir / f"graph_{track}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out, n, n_temporal, n_similar


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", nargs="?", help="index, track id, or path to a .pt")
    parser.add_argument("--all", action="store_true", help="render every sample graph")
    parser.add_argument("--out", default=None, help="output directory")
    args = parser.parse_args(argv)

    out_dir = Path(args.out) if args.out else project_root() / "results" / "plots" / "sample_graphs"

    if args.all:
        for path in _samples():
            out, n, t, s = render(path, out_dir)
            print(f"  {out.name:<34} {n:>3} nodes, {t:>3} temporal, {s:>3} similarity")
        print(f"\n{len(_samples())} graphs -> {out_dir}")
        return 0

    if args.target is None:
        samples = _samples()
        print(f"{len(samples)} sample graphs. Pass an index, a track id, or a path.\n")
        for i, path in enumerate(samples):
            print(f"  {i:>2}  {path.stem}")
        print("\n  python scripts/view_graph.py 0")
        print("  python scripts/view_graph.py --all")
        return 0

    path = resolve(args.target)
    if path is None:
        LOGGER.error("no graph matches %r; run without arguments to list them",
                     args.target)
        return 1
    out, n, t, s = render(path, out_dir)
    print(f"{out}\n  {n} nodes, {t} temporal edges, {s} similarity edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
