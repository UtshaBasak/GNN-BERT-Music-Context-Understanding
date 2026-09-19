#!/usr/bin/env python
"""the Task 1 sweep, written to run unattended on Kaggle.

Paste this into a Kaggle notebook cell (or add it as a utility script) after
attaching `kaggle_payload_task1.tar.gz` as a dataset, then use **Save & Run All**
so it executes in the background against the 12-hour limit.

    !tar xzf /kaggle/input/<your-dataset>/kaggle_payload_task1.tar.gz -C /kaggle/working
    %cd /kaggle/working
    !pip -q install torch-geometric          # ~11 s; required, see below
    !python scripts/kaggle_task1.py --model bert-base-uncased --epochs 8

**Internet must be ON** in the notebook sidebar (Kaggle disables it by default,
and enabling it needs a phone-verified account). `bert-base-uncased` is fetched
from HuggingFace at runtime, so without it the run dies at model load.

**Keep the torch-geometric line.** Task 1 needs no graphs, but `src.train`
imports the graph modules at module load, so the import fails without it. It is
a pure-Python wheel and installs in about 11 seconds once torch is present.

It runs the three freeze modes the report compares — `frozen_probe`, `top_n`,
`full_ft` — and, for the headline MusicCaps configuration, both `caption_masked`
and `caption_raw`, because the gap between those two *is* the leakage result.

Task 1 is text-only, so no graphs are needed and the payload is ~3 MB. Every run
checkpoints to `/kaggle/working` after each epoch, and results land in
`results/task1_seed{seed}_{tag}.json` so a killed session loses at most one run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dir, get_logger, project_root  # noqa: E402

LOGGER = get_logger("gbmc.kaggle_task1")

#: (freeze_mode, text_source, corpus, vocabulary) — the runs the report needs.
SWEEP = [
    ("frozen_probe", "caption_masked", "musiccaps", "musiccaps"),
    ("top_n", "caption_masked", "musiccaps", "musiccaps"),
    ("full_ft", "caption_masked", "musiccaps", "musiccaps"),
    # the leakage demonstration: identical to the headline but unmasked
    ("full_ft", "caption_raw", "musiccaps", "musiccaps"),
    # secondary, non-circular: MTAT metadata against the MTAT vocabulary
    ("full_ft", "metadata", "mtat", "mtat"),
]


def run_one(freeze_mode: str, text_source: str, corpus: str, vocab: str,
            model: str, epochs: int, seed: int, batch: int, dry_run: bool,
            workers: int = 2) -> dict:
    from src.train import main as train_main

    tag = f"{corpus}_{text_source}_{freeze_mode}"
    argv = [
        "--task", "1",
        "--seed", str(seed),
        "--device", "cuda",
        "--override",
        f"bert.model_name={model}",
        f"bert.freeze_mode={freeze_mode}",
        f"bert.freeze_epochs={0 if freeze_mode != 'top_n' else 2}",
        f"data.text_source={text_source}",
        f"data.tag_corpora=[{corpus}]",
        f"tags.vocab_source={vocab}",
        f"train.epochs={epochs}",
        f"train.batch_size={batch}",
        "train.grad_accum_steps=1",
        f"train.num_workers={workers}",
        "--run-tag", tag,
    ]
    LOGGER.info("=== %s ===", tag)
    if dry_run:
        print(" ".join(argv))
        return {"tag": tag, "argv": argv, "dry_run": True}

    train_main(argv)

    # --run-tag writes straight to the per-run filename, so nothing overwrites
    results = project_root() / "results"
    dst = results / f"task1_seed{seed}_{tag}.json"
    if dst.exists():
        payload = json.loads(dst.read_text(encoding="utf-8"))
        return {"tag": tag, "macro_f1": payload["test"].get("macro_f1"),
                "micro_f1": payload["test"].get("micro_f1"),
                "mean_auc_pr": payload["test"].get("mean_auc_pr"),
                "result_file": str(dst)}
    return {"tag": tag, "error": "no result file written"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Task 1 sweep for Kaggle.")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--only", nargs="*", default=None,
                        help="run only these freeze modes")
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader workers; 0 avoids CPU contention when "
                             "something else is running on the same machine")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    ensure_dir("results")
    sweep = [row for row in SWEEP
             if args.only is None or row[0] in args.only]

    summary = []
    for freeze_mode, text_source, corpus, vocab in sweep:
        try:
            summary.append(run_one(freeze_mode, text_source, corpus, vocab,
                                   args.model, args.epochs, args.seed,
                                   args.batch, args.dry_run, args.workers))
        except Exception as exc:
            LOGGER.error("%s/%s/%s failed: %s", corpus, text_source, freeze_mode, exc)
            summary.append({"tag": f"{corpus}_{text_source}_{freeze_mode}",
                            "error": f"{type(exc).__name__}: {exc}"})
        # write after every run: a 12-hour Kaggle session can end mid-sweep
        (project_root() / "results" / "task1_sweep.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))

    masked = next((r for r in summary if "caption_masked_full_ft" in r.get("tag", "")), None)
    raw = next((r for r in summary if "caption_raw_full_ft" in r.get("tag", "")), None)
    if masked and raw and masked.get("macro_f1") and raw.get("macro_f1"):
        gap = raw["macro_f1"] - masked["macro_f1"]
        print(f"\nLEAKAGE GAP: raw caption {raw['macro_f1']:.4f} vs masked "
              f"{masked['macro_f1']:.4f} = +{gap:.4f} macro-F1 of pure label leakage.")
        print("Report the masked number as Task 1; report the gap as the reason.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
