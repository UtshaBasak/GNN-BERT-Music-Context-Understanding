#!/usr/bin/env python
"""compress the qualitative material into single figures.

    python scripts/make_compact_figures.py [--what retrieval,cases]

The report is at ~8 pages of a 6-10 limit before the qualitative material lands.
Ten retrieval examples as ten text blocks, plus three case studies as three
figures, would overshoot on their own. The compression order was fixed in
advance so it is not decided under pressure:

1. ten retrieval examples -> **one** multi-panel figure
2. three case studies     -> **one** three-column figure with a shared caption
3. per-tag threshold detail -> appendix (already done)

The retrieval panel is not merely a smaller version of the text blocks. Plotting
the true match's rank on a log axis against the gallery size shows at a glance
which queries succeeded, which failed, and by how far -- which ten prose blocks
do not. The captions are still printed, truncated, so the reader can see what
kind of description each case involved.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dir, get_logger, project_root  # noqa: E402

LOGGER = get_logger("gbmc.figures")

OK_COLOUR = "#4f9d69"
FAIL_COLOUR = "#c1666b"
MID_COLOUR = "#d4a03c"


def retrieval_figure(examples_path: Path, out_path: Path, gallery: int | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(examples_path.read_text(encoding="utf-8"))
    examples = payload.get("examples", payload if isinstance(payload, list) else [])
    if not examples:
        LOGGER.warning("no retrieval examples in %s", examples_path)
        return None
    gallery = int(gallery or payload.get("gallery_size", 0) or 1)

    # The exporter now returns a larger pool: ten curated extremes for this
    # figure plus random draws for the listening study, which needs an unbiased
    # sample. Thirty rows would be fourteen inches tall, so the figure keeps the
    # curated ten it was designed for and the study keeps the random ones.
    curated = [e for e in examples if e.get("selection") in (None, "best", "worst")]
    examples = sorted(curated or examples,
                      key=lambda e: e.get("true_rank") or 10**6)[:12]
    n = len(examples)
    fig, ax = plt.subplots(figsize=(7.0, 0.46 * n + 1.4))

    ranks, labels, colours = [], [], []
    for example in examples:
        rank = int(example.get("true_rank") or gallery)
        ranks.append(max(rank, 1))
        caption = " ".join(str(example.get("query_caption", "")).split())
        labels.append(textwrap.shorten(caption, width=64, placeholder=" ..."))
        colours.append(OK_COLOUR if rank <= 3 else
                       MID_COLOUR if rank <= 10 else FAIL_COLOUR)

    y = np.arange(n)
    # Stems with an end marker rather than bars: on a log axis a bar for rank 1
    # spans from 1 to 1 and is invisible, which silently hides the best result.
    left = 0.88
    for i, (rank, colour) in enumerate(zip(ranks, colours)):
        ax.plot([left, rank], [i, i], color=colour, lw=3.2, solid_capstyle="butt")
        ax.plot([rank], [i], marker="o", ms=5.5, color=colour, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(n - 0.4, -1.3)                   # inverted, with room for the labels
    ax.set_xscale("log")
    ax.set_xlim(left, gallery * 1.5)
    ax.set_xlabel(f"rank of the true clip among {gallery:,} candidates "
                  "(log scale, lower is better)", fontsize=8)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # Guide lines labelled at the TOP. At the bottom they overlap the log-axis
    # tick labels -- only visible once the figure is actually rendered, which is
    # why these get looked at rather than assumed.
    ax.axvline(10, color="#999", lw=0.8, ls="--")
    ax.text(10, -0.7, "R@10", fontsize=7, color="#666", va="bottom", ha="center")
    ax.axvline(gallery / 2, color="#999", lw=0.8, ls="--")
    ax.text(gallery / 2, -0.7, "chance median", fontsize=7, color="#666",
            va="bottom", ha="center")

    for i, rank in enumerate(ranks):
        ax.text(rank * 1.25, i, str(rank), va="center", fontsize=7, color="#333")

    handles = [plt.Line2D([], [], color=c, lw=3.2, marker="o", ms=5.5) for c in
               (OK_COLOUR, MID_COLOUR, FAIL_COLOUR)]
    # "outside top-10", not "failure": rank 12 of 2,503 is far above chance and
    # calling it a failure would overstate what the colour means.
    ax.legend(handles, ["top-3", "top-10", "outside top-10"], fontsize=7,
              loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3,
              frameon=False)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    n_fail = sum(1 for r in ranks if r > 10)
    LOGGER.info("wrote %s (%d examples, %d failures, gallery %d)",
                out_path, n, n_fail, gallery)
    return {"path": str(out_path), "n_examples": n, "n_failures": n_fail,
            "gallery_size": gallery,
            "median_rank": float(np.median(ranks))}


def case_study_figure(case_dir: Path, out_path: Path):
    """Three case studies side by side, sharing one caption."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    panels = sorted(case_dir.glob("case_study_*.png"))[:3]
    if not panels:
        LOGGER.warning("no case-study panels in %s", case_dir)
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 2.8), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        ax.imshow(mpimg.imread(panel))
        ax.set_axis_off()
        ax.set_title(panel.stem.replace("case_study_", "case ").replace("_", " "),
                     fontsize=8)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s (%d panels)", out_path, len(panels))
    return {"path": str(out_path), "n_panels": len(panels)}


def _stack(paths, out_path, layout, figsize, titles=None, title_size=8):
    """Composite existing PNGs into one figure.

    D4 wants t-SNE, F1 curves, attention examples and S_graph in the report.
    They were all rendered from real runs during Phase C; what the page budget
    cannot take is nine more single-panel floats. Recombining beats
    regenerating: the pixels are already correct and re-running the t-SNE would
    move the embedding for no gain.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    paths = [Path(p) for p in paths]
    missing = [p for p in paths if not p.exists()]
    if missing:
        LOGGER.warning("missing panels, skipping %s: %s", out_path.name,
                       ", ".join(p.name for p in missing))
        return None

    rows, cols = layout
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat:
        ax.set_axis_off()
    for i, (ax, path) in enumerate(zip(flat, paths)):
        ax.imshow(mpimg.imread(path))
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=title_size)
    fig.tight_layout(pad=0.4)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s (%d panels)", out_path, len(paths))
    return {"path": str(out_path), "n_panels": len(paths)}


def tsne_figure(plots: Path):
    """Three t-SNE panels side by side, identical perplexity."""
    return _stack(
        [plots / "tsne_genre.png", plots / "tsne_mood.png",
         plots / "tsne_mood_mtat.png"],
        plots / "tsne_panels.png", layout=(1, 3), figsize=(7.2, 2.5),
        titles=["(a) FMA genre", "(b) DEAM quadrants", "(c) MTAT mood tags"])


def attention_figure(plots: Path):
    """The five Task 1 attention examples, stacked."""
    return _stack(
        [plots / f"bert_attention_0{i}.png" for i in range(5)],
        plots / "attention_examples.png", layout=(5, 1), figsize=(6.4, 8.0))


def diagnostics_figure(plots: Path):
    """F1-vs-epoch beside the S_graph structural control."""
    return _stack(
        [plots / "f1_vs_epoch.png", plots / "graph_coherence.png"],
        plots / "diagnostics.png", layout=(1, 2), figsize=(7.2, 2.7),
        titles=["(a) macro/micro-F1 vs epoch", "(b) $S_{graph}$ real vs rewired"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what",
                        default="retrieval,cases,tsne,attention,diagnostics")
    args = parser.parse_args(argv)

    root = project_root()
    plots = root / "results" / "plots"
    wanted = {w.strip() for w in args.what.split(",")}
    produced = {}

    if "retrieval" in wanted:
        path = root / "results" / "retrieval_examples" / "retrieval_examples.json"
        if path.exists():
            produced["retrieval"] = retrieval_figure(
                path, plots / "retrieval_examples.png")
        else:
            LOGGER.warning("%s not found -- run Task 4 and src.evaluate first", path)

    if "cases" in wanted:
        produced["cases"] = case_study_figure(plots, plots / "case_studies.png")

    if "tsne" in wanted:
        produced["tsne"] = tsne_figure(plots)
    if "attention" in wanted:
        produced["attention"] = attention_figure(plots)
    if "diagnostics" in wanted:
        produced["diagnostics"] = diagnostics_figure(plots)

    if not any(produced.values()):
        LOGGER.warning("nothing produced; the inputs do not exist yet")
        return 1
    for name, info in produced.items():
        if info:
            print(f"{name}: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
