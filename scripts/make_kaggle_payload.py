#!/usr/bin/env python
"""build the upload archive for Kaggle.

    python scripts/make_kaggle_payload.py [--out kaggle_payload.tar.gz]

**Graphs and text only. Never the mel caches.** The mel caches exist to feed the
CNN baseline, which runs locally in minutes; uploading them would multiply the
archive by an order of magnitude for no benefit and, on a metered connection,
would be the single slowest step in the project.

What goes in:

* ``data/processed/graphs/``          pre-built ``.pt`` graphs (the actual input)
* ``data/processed/norm_stats_*.json`` train-split statistics, so Kaggle applies
  exactly the same normalisation and cannot silently recompute it on the wrong split
* ``data/splits/``                    manifests, tag vocabulary, text variants
* ``config.yaml``, ``src/`` and ``scripts/``  so the remote run is the same code
  and has its entry point

Refuses to build if it would ship anything synthetic, and prints the size so the
figure can go straight into the session log.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (  # noqa: E402
    SYNTHETIC,
    detect_provenance,
    get_logger,
    load_config,
    project_root,
    resolve_path,
    save_json,
)

LOGGER = get_logger("gbmc.kaggle")

#: Never, under any flag. Mel caches are 5.8 GB and feed a baseline that runs
#: locally in minutes; audio and checkpoints have no business on a remote host.
EXCLUDE_ALWAYS = ("mels_", "mels_full", "_synthetic_smoke", "/synthetic/",
                  "\\synthetic\\", ".venv", "__pycache__", ".pt.tmp")
EXCLUDE_SUFFIXES_ALWAYS = (".mp3", ".wav", ".m4a", ".flac", ".ckpt", ".pt",
                           ".pth", ".safetensors", ".pyc")

#: Excluded by default, admitted by --include-features. Tasks 2-4 build their
#: graphs on the fly from these caches -- DataBundle sets graph_dir=None for
#: real data -- so a Task 3 payload without them cannot load a single batch.
#: Task 1 is text-only and does not need them, which is why they are opt-in.
EXCLUDE_UNLESS_FEATURES = ("features_",)
EXCLUDE_SUFFIXES_UNLESS_FEATURES = (".h5", ".hdf5")


def _is_excluded(path: Path, root: Path, include_features: bool = False) -> bool:
    rel = str(path.relative_to(root)).replace("\\", "/")
    marked = f"/{rel}"
    if path.suffix.lower() in EXCLUDE_SUFFIXES_ALWAYS:
        return True
    if any(token.replace("\\", "/") in marked for token in EXCLUDE_ALWAYS):
        return True
    if not include_features:
        if path.suffix.lower() in EXCLUDE_SUFFIXES_UNLESS_FEATURES:
            return True
        if any(token in marked for token in EXCLUDE_UNLESS_FEATURES):
            return True
    return False


def collect(root: Path, cfg, include_graphs: bool = True,
            include_features: bool = False) -> list[Path]:
    """The exact file list, with the exclusions applied."""
    processed = resolve_path(cfg["paths"]["processed"])
    splits = resolve_path(cfg["paths"]["splits"])

    wanted: list[Path] = []
    if include_features:
        # what Tasks 2-4 actually read; the .pt graphs below are exports for
        # inspection and are not on the training path for real data
        wanted += sorted(processed.glob("features_*.h5"))
        wanted += sorted(processed.glob("features_*.h5.keys.json"))
    if include_graphs and (processed / "graphs").exists():
        wanted += sorted((processed / "graphs").rglob("*.pt"))
        wanted += sorted((processed / "graphs").rglob("*.json"))
    wanted += sorted(processed.glob("norm_stats*.json"))
    if splits.exists():
        wanted += sorted(splits.glob("*.csv")) + sorted(splits.glob("*.json"))
    wanted.append(root / "config.yaml")
    wanted += sorted((root / "src").rglob("*.py"))
    # scripts/ too: kaggle_task1.py is the entry point the remote run executes,
    # and kaggle_setup.py stages the payload. Shipping src/ without scripts/
    # produces a payload that cannot actually be run.
    wanted += sorted((root / "scripts").rglob("*.py"))
    wanted.append(root / "requirements.txt")

    return [p for p in wanted if p.exists()
            and not _is_excluded(p, root, include_features)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the Kaggle upload archive.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="kaggle_payload.tar.gz")
    parser.add_argument("--no-graphs", action="store_true",
                        help="metadata and code only (graphs not built yet)")
    parser.add_argument("--include-features", action="store_true",
                        help="ship features_*.h5, which Tasks 2-4 read to build "
                             "graphs on the fly. Never ships mel caches.")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    root = project_root()
    files = collect(root, cfg, include_graphs=not args.no_graphs,
                    include_features=args.include_features)
    if not files:
        LOGGER.error("nothing to package")
        return 1

    # never ship fake data to a machine where it will be trained on
    for path in files:
        if path.suffix == ".csv" and "manifest" in path.name:
            import pandas as pd

            try:
                if detect_provenance(pd.read_csv(path)) == SYNTHETIC:
                    if not args.allow_synthetic:
                        LOGGER.error("%s is synthetic; refusing to package it", path.name)
                        return 2
            except Exception:
                pass
        if detect_provenance(path) == SYNTHETIC and not args.allow_synthetic:
            LOGGER.error("%s looks synthetic; refusing to package it", path)
            return 2

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    total_bytes = sum(p.stat().st_size for p in files)
    LOGGER.info("packaging %d files (%.1f MB uncompressed) -> %s",
                len(files), total_bytes / 1024**2, out)

    with tarfile.open(out, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=str(path.relative_to(root)).replace("\\", "/"))

    size_mb = out.stat().st_size / 1024**2
    by_kind: dict[str, int] = {}
    for path in files:
        by_kind[path.suffix or "(none)"] = by_kind.get(path.suffix or "(none)", 0) + 1

    manifest = {
        "archive": str(out),
        "size_mb": round(size_mb, 1),
        "uncompressed_mb": round(total_bytes / 1024**2, 1),
        "n_files": len(files),
        "files_by_extension": dict(sorted(by_kind.items())),
        "includes_features": bool(args.include_features),
        "excluded": ("mel caches, audio, checkpoints, synthetic artifacts"
                     + ("" if args.include_features else ", feature caches")),
    }
    stem = Path(out).stem.replace(".tar", "")
    save_json(manifest, root / "state" / f"{stem}.json")
    print(json.dumps(manifest, indent=2))
    LOGGER.info("archive is %.1f MB -- record this in the session log", size_mb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
