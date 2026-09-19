#!/usr/bin/env python
"""the seven-mode Task 3 fusion ablation, sharded across Kaggle GPUs.

    python scripts/kaggle_ablation.py --shard 0 --shards 2
    python scripts/kaggle_ablation.py --shards 2 --dry-run     # check the split

Twenty-one runs: seven fusion modes times seeds 42 / 1337 / 2024, all on MTAT,
all with `distilbert-base-uncased` and an epoch budget held **identical across
every mode** -- that identity is the entire point of an ablation, so it is a
single flag rather than something a caller can vary per run.

**Why this is not run locally.** bert-base measures ~68 ms/row on the local
GTX 1650, so twenty-one runs at full budget is roughly 26 hours. That would have
forced cutting rows out of the ablation table. A Kaggle session with two T4s runs
two shards concurrently and leaves the local GPU free for C5-C7, so nothing is
cut and nothing is queued behind it.

**Sharding is deterministic.** The run list is built in a fixed order and shard
*i* takes every *N*th entry from offset *i*. Any set of shards covering
``0..N-1`` therefore covers all twenty-one runs exactly once, with no overlap and
no coordination between the processes -- which matters because the two shards
run in separate processes that cannot see each other's progress.

**Resumable.** A run whose result JSON already exists is skipped, so a session
that dies at hour three restarts where it stopped rather than from the beginning.

Per-run wall-clock is printed so the T4 can be compared against the local
measurement and the remaining estimate corrected.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fusion_model import FUSION_MODES  # noqa: E402
from src.utils import ensure_dir, get_logger, project_root, save_json  # noqa: E402

LOGGER = get_logger("gbmc.kaggle_ablation")

SEEDS = (42, 1337, 2024)
CORPUS = "mtat"

#: MTAT rows, for the ms/row figure that makes T4 and local comparable
MTAT_TRAIN_ROWS = 16881


def build_runs(modes=FUSION_MODES, seeds=SEEDS) -> list[dict]:
    """The full run list, in a fixed order.

    Ordered seed-major so that if a session is cut short, the shards have
    covered whole seeds rather than leaving every mode with a ragged number of
    them -- an ablation table with three seeds for four modes and one for the
    rest is harder to report than one with fewer modes at full depth.
    """
    return [{"index": i, "mode": mode, "seed": seed,
             "run_tag": f"{CORPUS}_{mode}"}
            for i, (seed, mode) in enumerate(
                (s, m) for s in seeds for m in modes)]


def shard_of(runs: list[dict], shard: int, shards: int) -> list[dict]:
    """Every ``shards``-th run from offset ``shard``."""
    if not 0 <= shard < shards:
        raise ValueError(f"shard {shard} out of range for {shards} shards")
    return [r for r in runs if r["index"] % shards == shard]


def result_path(run: dict) -> Path:
    return (project_root() / "results"
            / f"task3_seed{run['seed']}_{run['run_tag']}.json")


def already_done(run: dict) -> bool:
    """True when a matching result exists -- the resume check."""
    path = result_path(run)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("%s is unreadable; re-running", path.name)
        return False
    if payload.get("provenance") == "synthetic":
        LOGGER.warning("%s is synthetic; re-running", path.name)
        return False
    if payload.get("fusion_mode") not in (None, run["mode"]):
        LOGGER.warning("%s records mode %s, expected %s; re-running",
                       path.name, payload.get("fusion_mode"), run["mode"])
        return False
    return payload.get("test", {}).get("macro_f1") is not None


def run_one(run: dict, model: str, epochs: int, batch: int, accum: int,
            device: str) -> dict:
    from src.train import main as train_main

    argv = [
        "--task", "3",
        "--seed", str(run["seed"]),
        "--device", device,
        "--num-workers", "0",
        "--run-tag", run["run_tag"],
        "--override",
        f"fusion.mode={run['mode']}",
        f"bert.model_name={model}",
        f"data.tag_corpora=[{CORPUS}]",
        "tags.vocab_source=mtat",
        "data.text_source=metadata",
        f"train.epochs={epochs}",
        f"train.batch_size={batch}",
        f"train.grad_accum_steps={accum}",
    ]
    started = time.time()
    train_main(argv)
    elapsed = time.time() - started

    path = result_path(run)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    test = payload.get("test", {})
    epochs_run = int(payload.get("epochs_run", 0) or 1)
    return {
        "mode": run["mode"], "seed": run["seed"], "index": run["index"],
        "macro_f1": test.get("macro_f1"),
        "macro_f1_fixed_half": test.get("macro_f1_fixed_half"),
        "valence_mae": test.get("valence_mae"),
        "arousal_r2": test.get("arousal_r2"),
        "epochs_run": epochs_run,
        "wall_clock_s": round(elapsed, 1),
        "seconds_per_epoch": round(elapsed / max(epochs_run, 1), 1),
        "ms_per_row": round(1000 * elapsed / max(epochs_run, 1) / MTAT_TRAIN_ROWS, 2),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--ablation-epochs", type=int, default=6,
                        help="held identical across every mode; that identity "
                             "is what makes the table an ablation")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modes", default=",".join(FUSION_MODES))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    for mode in modes:
        if mode not in FUSION_MODES:
            raise SystemExit(f"unknown fusion mode {mode!r}")

    runs = build_runs(modes, seeds)

    if args.dry_run:
        print(f"{len(runs)} runs total, {args.shards} shards\n")
        covered: dict[int, list[int]] = {}
        for shard in range(args.shards):
            mine = shard_of(runs, shard, args.shards)
            print(f"--- shard {shard}: {len(mine)} runs ---")
            for run in mine:
                print(f"  [{run['index']:>2}] seed {run['seed']:<5} {run['mode']}")
                covered.setdefault(run["index"], []).append(shard)
            print()
        missing = [r["index"] for r in runs if r["index"] not in covered]
        duplicated = {i: s for i, s in covered.items() if len(s) > 1}
        print(f"coverage: {len(covered)}/{len(runs)} runs assigned")
        if missing:
            print(f"FAIL: {len(missing)} run(s) assigned to no shard: {missing}")
        if duplicated:
            print(f"FAIL: run(s) in more than one shard: {duplicated}")
        if not missing and not duplicated:
            print("OK: every run appears in exactly one shard, no overlap")
        return 1 if (missing or duplicated) else 0

    mine = shard_of(runs, args.shard, args.shards)
    ensure_dir("results")
    out_path = (project_root() / "results"
                / f"kaggle_ablation_shard{args.shard}of{args.shards}.json")

    LOGGER.info("shard %d/%d: %d run(s), %s, %d epochs each, device %s "
                "(CUDA_VISIBLE_DEVICES=%s)",
                args.shard, args.shards, len(mine), args.model,
                args.ablation_epochs, args.device,
                os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))

    done, summary = [], []
    started = time.time()
    for position, run in enumerate(mine, start=1):
        label = f"{run['mode']} seed {run['seed']}"
        if already_done(run):
            LOGGER.info("[%d/%d] %s already done; skipping", position, len(mine), label)
            done.append(run["index"])
            continue
        LOGGER.info("[%d/%d] === %s ===", position, len(mine), label)
        try:
            row = run_one(run, args.model, args.ablation_epochs, args.batch,
                          args.accum, args.device)
            summary.append(row)
            LOGGER.info("[%d/%d] %s: macro-F1 %s in %.0f s (%.1f s/epoch, "
                        "%.2f ms/row)", position, len(mine), label,
                        f"{row['macro_f1']:.4f}" if row["macro_f1"] else "n/a",
                        row["wall_clock_s"], row["seconds_per_epoch"],
                        row["ms_per_row"])
        except Exception as exc:                                   # noqa: BLE001
            import traceback
            LOGGER.error("[%d/%d] %s FAILED: %s", position, len(mine), label, exc)
            traceback.print_exc()
            summary.append({"mode": run["mode"], "seed": run["seed"],
                            "index": run["index"],
                            "error": f"{type(exc).__name__}: {exc}"})
        # written after every run: a killed session keeps what finished
        save_json({"shard": args.shard, "shards": args.shards,
                   "model": args.model, "ablation_epochs": args.ablation_epochs,
                   "corpus": CORPUS, "n_assigned": len(mine),
                   "elapsed_s": round(time.time() - started, 1),
                   "runs": summary}, out_path)

    elapsed = time.time() - started
    finished = [r for r in summary if "error" in r or r.get("macro_f1") is not None]
    rates = [r["ms_per_row"] for r in summary if r.get("ms_per_row")]

    print(f"\nshard {args.shard}/{args.shards}: {len(finished)} run(s) in "
          f"{elapsed / 60:.1f} min ({len(done)} skipped as already done)")
    if rates:
        mean_rate = sum(rates) / len(rates)
        print(f"throughput: {mean_rate:.2f} ms/row against 68 ms/row measured "
              f"locally on a GTX 1650 -- {68 / mean_rate:.1f}x faster")
        remaining = len(runs) - len(done) - len(finished)
        if remaining > 0:
            per_run = elapsed / max(len(finished), 1)
            print(f"~{remaining * per_run / 3600:.1f} h of runs remain across "
                  "all shards at this rate")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
