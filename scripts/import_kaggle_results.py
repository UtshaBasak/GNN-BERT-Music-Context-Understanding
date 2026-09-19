#!/usr/bin/env python
"""Merge a finished Kaggle run back into this project.

    python scripts/import_kaggle_results.py <downloaded.zip | extracted_dir>

Kaggle hands you the notebook's `/kaggle/working` as a zip. This copies the
Task 1 result JSONs (and any checkpoints you chose to keep) into `results/`,
then prints the comparison table the report needs — including the leakage gap
between the masked and raw caption runs, which is the point of running both.

Three things are checked rather than assumed:

* **provenance** — a result produced from synthetic data is refused, so a stray
  smoke-test artifact cannot end up in the report;
* **threshold discipline** — every imported run must record `threshold_source:
  val`, because a number tuned on test is not reportable;
* **collisions** — an existing local result is never silently overwritten; pass
  `--force` if replacing it is genuinely what you want.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (  # noqa: E402
    SYNTHETIC,
    detect_provenance,
    ensure_dir,
    get_logger,
    load_json,
    project_root,
    save_json,
    vocabulary_hash,
)

LOGGER = get_logger("gbmc.import_kaggle")


def _extract(source: Path, workdir: Path) -> Path:
    if source.is_dir():
        return source
    if source.suffix.lower() == ".zip":
        out = ensure_dir(workdir / "kaggle_unzipped")
        with zipfile.ZipFile(source) as zf:
            zf.extractall(out)
        LOGGER.info("extracted %s -> %s", source.name, out)
        return out
    raise ValueError(f"expected a directory or a .zip, got {source}")


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("skipping unreadable %s: %s", path.name, exc)
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Import Kaggle results.")
    parser.add_argument("source", help="downloaded .zip or an extracted directory")
    parser.add_argument("--force", action="store_true",
                        help="overwrite result files that already exist locally")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--copy-checkpoints", action="store_true",
                        help="also copy any .pt checkpoints (large)")
    args = parser.parse_args(argv)

    root = project_root()
    results_dir = ensure_dir(root / "results")
    source = _extract(Path(args.source).expanduser(), root / "state")

    # the label spaces this project currently uses; anything scored against
    # something else is not comparable with the local results and is refused
    expected_hashes = {}
    for name, source_name in (("mtat", "tag_vocab.json"),
                              ("musiccaps", "musiccaps_tag_vocab.json")):
        path = root / "data" / "splits" / source_name
        if path.exists():
            payload = load_json(path)
            tags = payload["tags"] if isinstance(payload, dict) else payload
            expected_hashes[vocabulary_hash(tags)] = name
    LOGGER.info("accepting results against %d known vocabulary/vocabularies: %s",
                len(expected_hashes),
                ", ".join(f"{v}={k}" for k, v in expected_hashes.items()) or "none")

    candidates = sorted(source.rglob("task*_seed*.json")) + sorted(source.rglob("task1_sweep.json"))
    if not candidates:
        LOGGER.error("no task*_seed*.json found under %s -- is this the notebook output?", source)
        return 1

    imported, skipped, rejected = [], [], []
    for path in candidates:
        payload = _load(path)
        if payload is None:
            continue

        if path.name != "task1_sweep.json":
            if detect_provenance(payload) == SYNTHETIC and not args.allow_synthetic:
                LOGGER.error("REFUSED %s: provenance is synthetic", path.name)
                rejected.append((path.name, "synthetic provenance"))
                continue
            source_of_thresholds = payload.get("threshold_source")
            if source_of_thresholds not in (None, "val"):
                LOGGER.error("REFUSED %s: thresholds came from %r, not val",
                             path.name, source_of_thresholds)
                rejected.append((path.name, f"thresholds from {source_of_thresholds}"))
                continue

            # a result computed elsewhere is only comparable if it was
            # scored against a label space this repository still uses. A7.3
            # changed 7 of the 50 MusicCaps tags, so a run from before that is
            # numerically fine and semantically incompatible -- and nothing
            # about the file would reveal it.
            incoming = payload.get("tag_vocab_hash")
            if incoming is None and payload.get("tag_vocab"):
                incoming = vocabulary_hash(payload["tag_vocab"])
            if expected_hashes and incoming and incoming not in expected_hashes:
                LOGGER.error("REFUSED %s: vocabulary hash %s matches no current "
                             "vocabulary (known: %s)", path.name, incoming,
                             ", ".join(sorted(expected_hashes)))
                rejected.append((path.name, f"vocabulary hash {incoming} unknown"))
                continue
            if expected_hashes and not incoming:
                LOGGER.error("REFUSED %s: records no vocabulary, so it cannot be "
                             "shown comparable with the local results",
                             path.name)
                rejected.append((path.name, "no vocabulary recorded"))
                continue

        target = results_dir / path.name
        if target.exists() and not args.force:
            LOGGER.warning("SKIP %s: already exists locally (pass --force to replace)",
                           path.name)
            skipped.append(path.name)
            continue
        shutil.copy2(path, target)
        imported.append((path.name, payload))
        LOGGER.info("imported %s", path.name)

    if args.copy_checkpoints:
        ckpt_dir = ensure_dir(results_dir / "checkpoints")
        for path in sorted(source.rglob("*.pt")):
            shutil.copy2(path, ckpt_dir / path.name)
            LOGGER.info("imported checkpoint %s", path.name)

    # ---- the table the report needs ---------------------------------------- #
    runs = [(n, p) for n, p in imported if n != "task1_sweep.json" and "test" in p]
    if runs:
        print(f"\n{'run':<44}{'macro-F1':>10}{'micro-F1':>10}{'AUC-PR':>9}{'rows':>7}")
        print("-" * 80)
        for name, payload in sorted(runs, key=lambda r: -(r[1]["test"].get("macro_f1") or 0)):
            t = payload["test"]
            print(f"{name.replace('task1_seed42_', '').replace('.json', ''):<44}"
                  f"{t.get('macro_f1', float('nan')):>10.4f}"
                  f"{t.get('micro_f1', float('nan')):>10.4f}"
                  f"{t.get('mean_auc_pr', float('nan')):>9.4f}"
                  f"{t.get('n_rows', 0):>7}")

        masked = next((p for n, p in runs if "caption_masked_full_ft" in n), None)
        raw = next((p for n, p in runs if "caption_raw_full_ft" in n), None)
        if masked and raw:
            gap = raw["test"]["macro_f1"] - masked["test"]["macro_f1"]
            print(f"\nLEAKAGE GAP: raw caption {raw['test']['macro_f1']:.4f} vs masked "
                  f"{masked['test']['macro_f1']:.4f} = {gap:+.4f} macro-F1.")
            print("Report the MASKED number as Task 1. The gap is the result about leakage.")

    summary = {
        "source": str(source),
        "imported": [n for n, _ in imported],
        "skipped_existing": skipped,
        "rejected": rejected,
    }
    save_json(summary, root / "state" / "kaggle_import.json")
    print(f"\nimported {len(imported)}, skipped {len(skipped)}, rejected {len(rejected)}")
    if rejected:
        print("REJECTED:")
        for name, why in rejected:
            print(f"  {name}: {why}")
    print("\nNext: python -m src.evaluate --device cuda   (regenerates tables and plots)")
    return 0 if not rejected else 2


if __name__ == "__main__":
    raise SystemExit(main())
