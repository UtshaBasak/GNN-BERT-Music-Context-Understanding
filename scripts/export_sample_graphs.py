#!/usr/bin/env python
"""Export >= 20 sample graphs to ``data/processed/sample_graphs/``.

This directory is the one exception to the
"no binaries in git" rule -- see ``.gitignore``. Each ``.pt`` ships with a
``.json`` summary so a reader can check the data contract without loading torch.

    python scripts/export_sample_graphs.py [--synthetic] [--n 20] [--render 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.graph_builder import export_sample_graphs, visualise_graph  # noqa: E402
from src.utils import (  # noqa: E402
    ensure_dir,
    get_logger,
    guard_against_synthetic,
    load_config,
    resolve_path,
)

LOGGER = get_logger("gbmc.samples")


def _source(cfg, synthetic: bool, limit: int):
    """Prefer real cached graphs; fall back to synthetic; then to the raw manifest."""
    if synthetic:
        root = resolve_path(cfg["synthetic"]["out_dir"])
        graphs = root / "graphs"
        if not graphs.exists():
            raise FileNotFoundError(
                f"{graphs} not found -- run `python -m src.synthetic` first"
            )
        files = sorted(graphs.glob("*.pt"))[:limit]
        return [torch.load(f, weights_only=False) for f in files], "synthetic"

    # Prefer the pre-built per-corpus graph directories, and spread the sample
    # across corpora: a deliverable drawn entirely from MTAT would show none of
    # the label-sentinel variety the contract is built around.
    graph_root = resolve_path(cfg["paths"]["processed"]) / "graphs"
    corpora = [d.name for d in sorted(graph_root.glob("*"))
               if (d / "segment").is_dir()] if graph_root.exists() else []
    if corpora:
        per_corpus = max(1, limit // max(len(corpora), 1))
        graphs = []
        for name in corpora:
            files = sorted((graph_root / name / "segment").glob("*.pt"))[:per_corpus]
            graphs += [torch.load(f, weights_only=False) for f in files]
        # top up from the largest corpus if integer division left us short
        if len(graphs) < limit and corpora:
            extra = sorted((graph_root / corpora[0] / "segment").glob("*.pt"))
            for f in extra[per_corpus:]:
                if len(graphs) >= limit:
                    break
                graphs.append(torch.load(f, weights_only=False))
        LOGGER.info("sampled %d graphs across %d corpora: %s",
                    len(graphs), len(corpora), ", ".join(corpora))
        return graphs[:limit], "real"

    from src.datasets import MusicGraphDataset

    splits_dir = resolve_path(cfg["paths"]["splits"])
    processed = resolve_path(cfg["paths"]["processed"])
    frames = [pd.read_csv(p) for p in sorted(splits_dir.glob("*_manifest.csv"))]
    caches = {p.stem.replace("features_", ""): p
              for p in processed.glob("features_*.h5")}
    if not frames or not caches:
        raise FileNotFoundError(
            "no feature caches in data/processed/ (or no manifests). "
            "Run `make splits && make features`, or pass --synthetic."
        )
    manifest = pd.concat(frames, ignore_index=True)
    stats = {}
    for name in caches:
        path = processed / f"norm_stats_{name}.json"
        if path.exists():
            stats[name] = json.loads(path.read_text(encoding="utf-8"))

    dataset = MusicGraphDataset(manifest, cfg=cfg, h5_path=caches, norm_stats=stats)
    graphs, i = [], 0
    while len(graphs) < limit and i < len(dataset):
        try:
            graphs.append(dataset[i])
        except KeyError:                 # not every manifest row is cached yet
            pass
        i += 1
    return graphs, "real"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export a sample of stored segment graphs.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out-dir", default="data/processed/sample_graphs")
    parser.add_argument("--render", type=int, default=5,
                        help="also write this many PNG renders")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    n = max(20, int(args.n))             # the deliverable requires at least 20
    graphs, source = _source(cfg, args.synthetic, n)
    # The committed sample graphs are the repository's runnable examples, so a
    # synthetic one slipping in is a submission failure, not a warning.
    guard_against_synthetic(graphs, args.synthetic, "the graphs being exported")
    if len(graphs) < 20:
        LOGGER.warning("only %d graphs available; the deliverable wants >= 20", len(graphs))

    written = export_sample_graphs(graphs, args.out_dir, n=len(graphs))

    out = ensure_dir(args.out_dir)
    for i, data in enumerate(graphs[: int(args.render)]):
        visualise_graph(data, out_path=out / f"render_{i:02d}.png",
                        title=f"{getattr(data, 'dataset', source)} / "
                              f"{getattr(data, 'track_id', i)}")

    print(json.dumps({
        "source": source,
        "n_graphs": len(written),
        "out_dir": str(out),
        "renders": min(int(args.render), len(graphs)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
