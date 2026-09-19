#!/usr/bin/env python
"""render the FMA-small genre confusion matrix as a report figure.

    python scripts/plot_genre_confusion.py

Reads whichever genre runs exist in ``results/`` (the GNN's
``task2_seed*_fma_genre.json`` and B2's entry in ``baselines_seed*.json``) and
writes one panel per model to ``results/plots/genre_confusion.png``.

Row-normalised, because the interesting question is *what a class gets confused
with*, and raw counts answer that badly whenever the classes are not exactly
equal -- FMA-small's test split is 100 clips per class except Pop at 101.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dir, get_logger, load_config, project_root  # noqa: E402

LOGGER = get_logger("gbmc.genreplot")


def _panels(results: Path) -> list[tuple[str, np.ndarray, list, float]]:
    """``(title, confusion, class_names, accuracy)`` for every genre run found."""
    found = []
    for path in sorted(results.glob("task2_seed*_fma_genre.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        test = payload.get("test", {})
        if "genre_confusion" not in test:
            continue
        found.append((
            f"T2 GNN  (acc {test['genre_accuracy']:.3f})",
            np.asarray(test["genre_confusion"], dtype=float),
            payload.get("genre_names") or payload.get("tag_vocab") or [],
            float(test["genre_accuracy"]),
        ))

    for path in sorted(results.glob("baselines_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload.get("baselines", []):
            if "genre_confusion" not in entry:
                continue
            found.append((
                f"B2 mel CNN  (acc {entry['genre_accuracy']:.3f})",
                np.asarray(entry["genre_confusion"], dtype=float),
                entry.get("genre_names") or [],
                float(entry["genre_accuracy"]),
            ))
    return found


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    results = project_root() / cfg["paths"]["results"]
    panels = _panels(results)
    if not panels:
        raise SystemExit("no genre runs with a confusion matrix in results/")

    names = cfg.get("data", {}).get("genre_names") or []
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.0),
                             squeeze=False)
    for ax, (title, cm, labels, _acc) in zip(axes[0], panels):
        labels = list(labels) or names or [str(i) for i in range(cm.shape[0])]
        rows = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, np.maximum(rows, 1e-9))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title, fontsize=10)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if norm[i, j] >= 0.01:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if norm[i, j] > 0.5 else "#333333")
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)

    fig.suptitle("FMA-small genre confusion, row-normalised (chance = 0.125)",
                 fontsize=11)
    fig.tight_layout()
    out = ensure_dir(results / "plots") / "genre_confusion.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s (%d panel(s))", out, len(panels))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
