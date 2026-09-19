#!/usr/bin/env python
"""how much do val-tuned decision thresholds actually move?

    python scripts/threshold_bootstrap.py [--n-boot 100] [--pattern "*_scores.npz"]

Every multi-label number in this project is reported at a per-tag threshold
fitted on the validation split. That fit is a fit like any other, and MTAT's
validation split is **977 clips drawn from 14 artists** -- small, and correlated
within artist on top of that. A single point estimate of "tuned macro-F1" hides
whatever variance the tuning step contributes.

So: resample the validation rows with replacement 100 times, re-tune all 50
thresholds on each replicate, and apply each threshold vector unchanged to the
*fixed* test split. What comes out is

* the per-tag standard deviation of the tuned threshold, and
* the spread of test macro-F1 that the tuning alone induces.

**The decision rule, fixed in advance:** if the test macro-F1 spread exceeds
0.02, the tuned number is not stable enough to stand alone and every table must
carry the fixed-0.5 number beside it. The verdict is written into the output so
it cannot be quietly reinterpreted later.

Reads the score matrices that `src.train` and `scripts/run_baselines.py` write
to `results/scores/`, and writes `results/threshold_bootstrap.json`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics as M  # noqa: E402
from src.utils import get_logger, load_config, project_root, save_json  # noqa: E402

LOGGER = get_logger("gbmc.threshold_boot")

#: above this spread in test macro-F1, tuned numbers must be reported alongside
#: the fixed-0.5 ones rather than on their own
SPREAD_LIMIT = 0.02


def analyse(path: Path, n_boot: int, seed: int) -> dict | None:
    payload = np.load(path, allow_pickle=True)
    needed = ("val_y_true", "val_y_score", "test_y_true", "test_y_score")
    if any(k not in payload for k in needed):
        LOGGER.warning("%s has no tag score matrices; skipping", path.name)
        return None
    if payload["val_y_true"].size == 0 or payload["test_y_true"].size == 0:
        LOGGER.info("%s is a single-label run (no thresholds to tune); skipping",
                    path.name)
        return None

    vocab = ([str(t) for t in payload["tag_vocab"]]
             if "tag_vocab" in payload else [])
    out = M.bootstrap_thresholds(
        payload["val_y_true"], payload["val_y_score"],
        payload["test_y_true"], payload["test_y_score"],
        n_boot=n_boot, seed=seed,
    )
    out["run"] = path.stem.replace("_scores", "")
    out["source"] = str(path.relative_to(project_root()))
    out["tag_vocab"] = vocab
    out["stable"] = bool(out["test_macro_f1_spread"] <= SPREAD_LIMIT)

    if vocab and len(vocab) == len(out["threshold_std_per_tag"]):
        order = np.argsort(-np.asarray(out["threshold_std_per_tag"]))
        out["least_stable_tags"] = [
            {"tag": vocab[i],
             "threshold_mean": round(out["threshold_mean_per_tag"][i], 4),
             "threshold_std": round(out["threshold_std_per_tag"][i], 4)}
            for i in order[:10]
        ]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pattern", default="*_scores.npz")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    scores_dir = project_root() / cfg["paths"]["results"] / "scores"
    if not scores_dir.exists():
        raise SystemExit(
            f"{scores_dir} does not exist -- train something first; every run "
            "writes its val/test score matrices there"
        )

    files = sorted(scores_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files matching {args.pattern} in {scores_dir}")

    runs = [r for r in (analyse(p, args.n_boot, args.seed) for p in files) if r]
    if not runs:
        raise SystemExit("no multi-label runs found to bootstrap")

    payload = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "spread_limit": SPREAD_LIMIT,
        "decision": (
            "report fixed-0.5 numbers alongside tuned ones in all tables"
            if any(not r["stable"] for r in runs)
            else "tuned thresholds are stable; tuned numbers may stand alone"
        ),
        "runs": runs,
    }
    out = save_json(payload, project_root() / cfg["paths"]["results"]
                    / "threshold_bootstrap.json")

    header = (f"{'run':<38}{'tuned':>8}{'fixed .5':>10}{'boot mean':>11}"
              f"{'spread':>9}{'thr sd':>8}  verdict")
    print(header)
    print("-" * len(header))
    for r in runs:
        print(f"{r['run'][:37]:<38}"
              f"{r['test_macro_f1_point']:>8.4f}"
              f"{r['test_macro_f1_fixed_half']:>10.4f}"
              f"{r['test_macro_f1_mean']:>11.4f}"
              f"{r['test_macro_f1_spread']:>9.4f}"
              f"{r['threshold_std_mean']:>8.3f}  "
              f"{'stable' if r['stable'] else 'UNSTABLE'}")
    print(f"\n{payload['decision']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
