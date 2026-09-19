#!/usr/bin/env python
"""does the graph topology actually carry signal?

    python scripts/structural_controls.py [--domains genre,tags] [--seeds 42]

Task 2's claim is that musical structure helps. Two controls test it, and both
are run on **both** Task 2 domains (FMA-small genre headline, MTAT tagging
secondary):

**Random rewiring.** Degree-preserving double-edge swaps destroy which segment
connects to which while leaving every node's degree, and therefore every
aggregation statistic, untouched. If the score survives, the GNN was never using
the topology -- it was a permutation-invariant pooled-feature MLP with extra
steps. That is a legitimate negative result and belongs in the report: a control
that fires is worth more than a headline number with no control.

**Edge-type ablation.** Temporal chain only, k-NN similarity only, or both. The
two edge kinds encode different things -- sequence versus repetition -- and if
one carries all the signal the other is decoration.

Read the results against **B4** (PCA + MLP on the same 96-dim features, mean
pooled, no structure at all), not against random. B4 is what isolates the
graph's contribution, and it already scores 0.3239 macro-F1 against the GNN's
~0.37, so the margin under test here was never going to be large.

Every run writes `results/task2_seed{S}_{domain}_{control}.json` and its score
matrices, so nothing overwrites the headline runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dir, get_logger, project_root  # noqa: E402

LOGGER = get_logger("gbmc.controls")

#: control name -> config overrides. "baseline" is re-run rather than reused so
#: every row in the table comes from one code version and one epoch budget.
CONTROLS = {
    "baseline": [],
    "rewired": ["graph.rewire=true"],
    "temporal_only": ["graph.similarity_edges=false"],
    "similarity_only": ["graph.temporal_edges=false"],
}

DOMAINS = {
    "genre": ["data.task2_target=genre"],
    "tags": ["data.task2_target=tags"],
}


def run_one(domain: str, control: str, seed: int, epochs: int, batch: int,
            dry_run: bool) -> dict:
    from src.train import main as train_main

    tag = f"{domain}_{control}"
    argv = [
        "--task", "2",
        "--seed", str(seed),
        "--device", "cuda",
        "--num-workers", "0",
        "--run-tag", tag,
        "--override",
        *DOMAINS[domain],
        *CONTROLS[control],
        f"train.epochs={epochs}",
        f"train.batch_size={batch}",
    ]
    LOGGER.info("=== %s seed %d ===", tag, seed)
    if dry_run:
        print(" ".join(argv))
        return {"tag": tag, "seed": seed, "dry_run": True}

    train_main(argv)

    path = project_root() / "results" / f"task2_seed{seed}_{tag}.json"
    if not path.exists():
        return {"tag": tag, "seed": seed, "error": "no result file"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    test = payload.get("test", {})
    return {
        "tag": tag, "domain": domain, "control": control, "seed": seed,
        "accuracy": test.get("genre_accuracy"),
        "macro_f1": test.get("genre_macro_f1", test.get("macro_f1")),
        "micro_f1": test.get("micro_f1"),
        "mean_auc_pr": test.get("mean_auc_pr"),
        "macro_f1_fixed_half": test.get("macro_f1_fixed_half"),
        "wall_clock_s": payload.get("wall_clock_s"),
        "epochs_run": payload.get("epochs_run"),
        "result_file": str(path.relative_to(project_root())),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", default="genre,tags")
    parser.add_argument("--controls", default=",".join(CONTROLS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    controls = [c.strip() for c in args.controls.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for name in domains:
        if name not in DOMAINS:
            raise SystemExit(f"unknown domain {name!r}; pick from {list(DOMAINS)}")
    for name in controls:
        if name not in CONTROLS:
            raise SystemExit(f"unknown control {name!r}; pick from {list(CONTROLS)}")

    ensure_dir("results")
    summary = []
    for seed in seeds:
        for domain in domains:
            for control in controls:
                try:
                    summary.append(run_one(domain, control, seed, args.epochs,
                                           args.batch, args.dry_run))
                except Exception as exc:                       # noqa: BLE001
                    LOGGER.error("%s/%s seed %d failed: %s", domain, control,
                                 seed, exc)
                    summary.append({"tag": f"{domain}_{control}", "seed": seed,
                                    "error": f"{type(exc).__name__}: {exc}"})
                # written after every run: a killed session keeps what finished
                (project_root() / "results" / "structural_controls.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8")

    if args.dry_run:
        return 0

    print(f"\n{'run':<34}{'seed':>6}{'acc':>9}{'macro-F1':>10}{'@0.5':>9}{'epochs':>8}")
    print("-" * 76)
    for row in summary:
        if "error" in row:
            print(f"{row['tag']:<34}{row['seed']:>6}   {row['error'][:40]}")
            continue
        acc = row.get("accuracy")
        fixed = row.get("macro_f1_fixed_half")
        print(f"{row['tag']:<34}{row['seed']:>6}"
              f"{('-' if acc is None else f'{acc:.4f}'):>9}"
              f"{row['macro_f1']:>10.4f}"
              f"{('-' if fixed is None else f'{fixed:.4f}'):>9}"
              f"{row.get('epochs_run', 0):>8}")

    # the only comparison that matters: did destroying the topology cost anything?
    for domain in domains:
        base = next((r for r in summary
                     if r.get("domain") == domain and r.get("control") == "baseline"
                     and r.get("macro_f1") is not None), None)
        rewired = next((r for r in summary
                        if r.get("domain") == domain and r.get("control") == "rewired"
                        and r.get("macro_f1") is not None), None)
        if base and rewired:
            drop = base["macro_f1"] - rewired["macro_f1"]
            verdict = ("the topology carries signal" if drop > 0.029 else
                       "WITHIN THE 0.0288 NOISE FLOOR -- the graph structure is "
                       "not measurably contributing")
            print(f"\n{domain}: rewiring costs {drop:+.4f} macro-F1 -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
