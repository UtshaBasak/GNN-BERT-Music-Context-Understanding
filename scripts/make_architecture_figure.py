#!/usr/bin/env python
"""the architecture diagram the report was missing.

    python scripts/make_architecture_figure.py

The Method section describes the model in equations, which is precise and hard
to hold in the head. The report had
none of the model itself -- every figure was a result. This draws the one
picture that makes the four tasks legible as variations on a shared
representation: which encoder each task uses, where the frozen node-feature
contract sits, and which paths carry a loss.

Drawn rather than photographed: no data is read, so it cannot go stale.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dir, get_logger, project_root  # noqa: E402

LOGGER = get_logger("gbmc.figures.arch")

AUDIO = "#8fb8de"
GRAPH = "#a8d5ba"
TEXT = "#e8c3a0"
FUSE = "#d4b8e0"
EDGE = "#5a5a5a"


def build(out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.set_axis_off()

    def box(x, y, w, h, label, colour, size=7.0, style="round,pad=0.35"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                    facecolor=colour, edgecolor=EDGE, linewidth=0.9))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=size, linespacing=1.35)

    def arrow(x1, y1, x2, y2, style="-|>", dashed=False):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9,
            linewidth=0.9, color=EDGE, linestyle="--" if dashed else "-",
            shrinkA=1, shrinkB=1))

    # ---- audio path ------------------------------------------------------ #
    box(1, 34, 15, 11, "audio\n22.05 kHz", AUDIO)
    box(19, 34, 17, 11, "segment\nwindows", AUDIO)
    box(39, 34, 19, 11, "segment graph\n$x\\in\\mathbb{R}^{96}$\n"
                        "$e=[\\mathrm{temporal},\\cos]$", GRAPH, size=6.4)
    box(61, 34, 16, 11, "GNN\nSAGE / GATv2\n2 layers", GRAPH, size=6.6)

    arrow(16, 39.5, 19, 39.5)
    arrow(36, 39.5, 39, 39.5)
    arrow(58, 39.5, 61, 39.5)

    # ---- text path ------------------------------------------------------- #
    box(1, 8, 15, 11, "text\ncaption / tags\n/ metadata", TEXT, size=6.4)
    box(19, 8, 17, 11, "BERT\ntokeniser", TEXT)
    box(39, 8, 19, 11, "BERT / DistilBERT\n$\\mathbf{H}_{\\mathrm{text}}$", TEXT, size=6.6)
    arrow(16, 13.5, 19, 13.5)
    arrow(36, 13.5, 39, 13.5)
    arrow(58, 13.5, 61, 21, style="-|>")

    # ---- fusion ---------------------------------------------------------- #
    box(61, 17, 16, 11, "fusion\ncross-attention\n$\\mathbf{z}$", FUSE, size=6.6)
    arrow(69, 34, 69, 28)

    # ---- heads ----------------------------------------------------------- #
    box(82, 38, 17, 8.5, "tags / genre\nBCE / CE", "#f2f2f2", size=6.4)
    box(82, 25, 17, 8.5, "valence, arousal\nMSE ($\\mathcal{L}_{\\mathrm{aux}}$)",
        "#f2f2f2", size=6.2)
    box(82, 8, 17, 8.5, "InfoNCE\nretrieval", "#f2f2f2", size=6.4)

    arrow(77, 39.5, 82, 42)        # gnn-only -> tags (task 2)
    arrow(77, 22.5, 82, 42)        # fusion -> tags (task 3)
    arrow(77, 22.5, 82, 29)        # fusion -> emotion (task 3)
    arrow(58, 13.5, 82, 12.5)      # text tower -> InfoNCE (task 4)
    arrow(58, 17.5, 82, 39, dashed=True)   # text-only -> tags (task 1)
    arrow(77, 36, 82, 12.5, dashed=True)   # graph tower -> InfoNCE

    # ---- task labels ----------------------------------------------------- #
    for x, y, t in [(69, 46.5, "Task 2: GNN only"),
                    (52, 31.5, "Task 3: fusion + multi-task"),
                    (48, 3.5, "Task 1: text only     Task 4: dual encoder")]:
        ax.text(x, y, t, ha="center", va="center", fontsize=6.6,
                style="italic", color="#333")

    ax.text(48.5, 48.5, "the 96-d node feature contract is frozen across all "
                        "four tasks", ha="center", va="center", fontsize=6.2,
            color="#666")

    fig.tight_layout(pad=0.2)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)
    return out_path


def main(argv=None) -> int:
    out = project_root() / "results" / "plots" / "architecture.png"
    build(out)
    print(f"architecture: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
