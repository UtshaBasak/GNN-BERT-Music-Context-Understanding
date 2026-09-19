#!/usr/bin/env python
"""the seven-mode Task 3 fusion ablation, run under the noise-floor rule.

    python scripts/fusion_ablation.py --corpus mtat --seeds 42,1337,2024
    python scripts/fusion_ablation.py --summary-only          # re-tabulate

**The rule this script exists to enforce.** A7.4 measured a tuned test macro-F1
spread of **0.0288** across 100 validation resamples on MTAT. Any two ablation
rows closer than that are indistinguishable from validation-split noise, and it
is entirely plausible that cross-attention, early concat and late concat all
land inside it. If they do, the finding is that *they are not separable at this
validation-set size* -- not that whichever printed highest is best. The summary
therefore prints a delta-vs-best column, marks every row inside the floor, and
refuses to name a winner when the top rows overlap.

**Two budgets, both documented.** The seven-mode sweep runs `distilbert-base-
uncased` at a reduced epoch count held identical across all modes, because seven
modes times three seeds is the longest job in the project. The headline table
then re-runs only the winning mode plus `gnn_only` and `bert_only` at full
budget with `bert-base-uncased`. Mixing the two budgets in one table would be
meaningless, so the budget is recorded in every result file.

Each run writes `results/task3_seed{S}_{corpus}_{mode}.json`.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fusion_model import FUSION_MODES  # noqa: E402
from src.utils import ensure_dir, get_logger, project_root, save_json  # noqa: E402

LOGGER = get_logger("gbmc.ablation")

#: 100 validation resamples on MTAT. Differences below this are noise.
NOISE_FLOOR = 0.0288

CORPORA = {
    "mtat": {
        "tag_corpora": "[mtat]",
        "vocab_source": "mtat",
        "text_source": "metadata",
    },
    "musiccaps": {
        "tag_corpora": "[musiccaps]",
        "vocab_source": "musiccaps",
        "text_source": "caption_masked",
    },
}


def run_one(mode: str, corpus: str, seed: int, model: str, epochs: int,
            batch: int, accum: int, budget: str, dry_run: bool) -> dict:
    from src.train import main as train_main

    spec = CORPORA[corpus]
    tag = f"{corpus}_{mode}"
    argv = [
        "--task", "3",
        "--seed", str(seed),
        "--device", "cuda",
        "--num-workers", "0",
        "--run-tag", tag,
        "--override",
        f"fusion.mode={mode}",
        f"bert.model_name={model}",
        f"data.tag_corpora={spec['tag_corpora']}",
        f"tags.vocab_source={spec['vocab_source']}",
        f"data.text_source={spec['text_source']}",
        f"train.epochs={epochs}",
        f"train.batch_size={batch}",
        f"train.grad_accum_steps={accum}",
    ]
    LOGGER.info("=== %s seed %d (%s budget) ===", tag, seed, budget)
    if dry_run:
        print(" ".join(argv))
        return {"tag": tag, "seed": seed, "dry_run": True}

    train_main(argv)

    path = project_root() / "results" / f"task3_seed{seed}_{tag}.json"
    if not path.exists():
        return {"tag": tag, "seed": seed, "error": "no result file"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    # stamp the budget so the two tables can never be silently merged
    payload["ablation_budget"] = budget
    payload["ablation_model"] = model
    save_json(payload, path)

    test = payload.get("test", {})
    return {
        "mode": mode, "corpus": corpus, "seed": seed, "budget": budget,
        "model": model,
        "macro_f1": test.get("macro_f1"),
        "macro_f1_fixed_half": test.get("macro_f1_fixed_half"),
        "micro_f1": test.get("micro_f1"),
        "mean_auc_pr": test.get("mean_auc_pr"),
        "valence_mae": test.get("valence_mae"),
        "arousal_mae": test.get("arousal_mae"),
        "valence_r2": test.get("valence_r2"),
        "arousal_r2": test.get("arousal_r2"),
        "wall_clock_s": payload.get("wall_clock_s"),
        "epochs_run": payload.get("epochs_run"),
    }


def _agg(values):
    clean = [v for v in values if isinstance(v, (int, float)) and v == v]
    if not clean:
        return None, None, 0
    sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return statistics.fmean(clean), sd, len(clean)


def summarise(rows: list[dict], corpus: str) -> dict:
    """Aggregate over seeds and apply the noise-floor rule."""
    per_mode = {}
    for mode in FUSION_MODES:
        subset = [r for r in rows if r.get("mode") == mode and r.get("corpus") == corpus]
        if not subset:
            continue
        tuned_mean, tuned_sd, n = _agg([r.get("macro_f1") for r in subset])
        fixed_mean, fixed_sd, _ = _agg([r.get("macro_f1_fixed_half") for r in subset])
        if tuned_mean is None:
            continue
        per_mode[mode] = {
            "mode": mode, "n_seeds": n,
            "macro_f1_mean": tuned_mean, "macro_f1_sd": tuned_sd,
            "fixed_half_mean": fixed_mean, "fixed_half_sd": fixed_sd,
            "seeds": sorted(r["seed"] for r in subset),
            "wall_clock_s": _agg([r.get("wall_clock_s") for r in subset])[0],
        }

    if not per_mode:
        return {}

    best = max(per_mode.values(), key=lambda r: r["macro_f1_mean"])
    for row in per_mode.values():
        row["delta_vs_best"] = row["macro_f1_mean"] - best["macro_f1_mean"]
        row["within_noise_of_best"] = abs(row["delta_vs_best"]) < NOISE_FLOOR

    tied = sorted((r["mode"] for r in per_mode.values() if r["within_noise_of_best"]))
    separable = len(tied) == 1
    return {
        "corpus": corpus,
        "noise_floor": NOISE_FLOOR,
        "best_mode": best["mode"] if separable else None,
        "modes_within_noise_of_best": tied,
        "separable": separable,
        "verdict": (
            f"{best['mode']} is the only mode outside the {NOISE_FLOOR} noise "
            "floor of the best; the ordering is reportable."
            if separable else
            f"{len(tied)} modes ({', '.join(tied)}) lie within the "
            f"{NOISE_FLOOR} noise floor of each other. They are NOT separable "
            "at this validation-set size, and no ordering among them may be "
            "claimed."
        ),
        "rows": [per_mode[m] for m in FUSION_MODES if m in per_mode],
    }


def print_table(summary: dict) -> None:
    if not summary:
        print("no rows to summarise")
        return
    print(f"\nTask 3 fusion ablation -- {summary['corpus']}")
    header = (f"{'mode':<18}{'macro-F1':>18}{'@0.5':>18}"
              f"{'delta':>9}{'seeds':>7}  note")
    print(header)
    print("-" * len(header))
    for row in summary["rows"]:
        tuned = f"{row['macro_f1_mean']:.4f} +-{row['macro_f1_sd']:.4f}"
        fixed = ("---" if row["fixed_half_mean"] is None
                 else f"{row['fixed_half_mean']:.4f} +-{row['fixed_half_sd']:.4f}")
        note = "within noise of best" if row["within_noise_of_best"] else ""
        print(f"{row['mode']:<18}{tuned:>18}{fixed:>18}"
              f"{row['delta_vs_best']:>+9.4f}{row['n_seeds']:>7}  {note}")
    print(f"\nnoise floor {summary['noise_floor']} macro-F1 "
          "(A7.4, 100 validation resamples)")
    print(summary["verdict"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="mtat", choices=sorted(CORPORA))
    parser.add_argument("--modes", default=",".join(FUSION_MODES))
    parser.add_argument("--seeds", default="42,1337,2024")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=6,
                        help="held identical across all modes; the whole point")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--budget", default="ablation",
                        help="'ablation' (distilbert, reduced) or 'headline'")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    out_path = project_root() / "results" / f"fusion_ablation_{args.corpus}.json"
    ensure_dir("results")

    if args.summary_only:
        if not out_path.exists():
            raise SystemExit(f"{out_path} does not exist yet")
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        print_table(summarise(payload.get("runs", []), args.corpus))
        return 0

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        if mode not in FUSION_MODES:
            raise SystemExit(f"unknown mode {mode!r}; pick from {list(FUSION_MODES)}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    runs = []
    if out_path.exists():                    # resume rather than restart
        runs = json.loads(out_path.read_text(encoding="utf-8")).get("runs", [])
        done = {(r.get("mode"), r.get("seed")) for r in runs}
        LOGGER.info("resuming: %d run(s) already recorded", len(done))
    else:
        done = set()

    for seed in seeds:
        for mode in modes:
            if (mode, seed) in done:
                LOGGER.info("skipping %s seed %d (already done)", mode, seed)
                continue
            try:
                runs.append(run_one(mode, args.corpus, seed, args.model,
                                    args.epochs, args.batch, args.accum,
                                    args.budget, args.dry_run))
            except Exception as exc:                          # noqa: BLE001
                LOGGER.error("%s seed %d failed: %s", mode, seed, exc)
                runs.append({"mode": mode, "corpus": args.corpus, "seed": seed,
                             "error": f"{type(exc).__name__}: {exc}"})
            if not args.dry_run:
                save_json({"corpus": args.corpus, "budget": args.budget,
                           "model": args.model, "epochs": args.epochs,
                           "noise_floor": NOISE_FLOOR, "runs": runs}, out_path)

    if args.dry_run:
        return 0

    summary = summarise(runs, args.corpus)
    save_json({"corpus": args.corpus, "budget": args.budget, "model": args.model,
               "epochs": args.epochs, "noise_floor": NOISE_FLOOR,
               "runs": runs, "summary": summary}, out_path)
    print_table(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
