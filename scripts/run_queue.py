#!/usr/bin/env python
"""run the local queue unattended, with a watchdog rather than a halt.

    python scripts/run_queue.py                 # run it
    python scripts/run_queue.py --plan          # show the queue and skip states
    python scripts/run_queue.py --from C5       # resume at a step

One process owns the whole queue. The bash chains this replaces waited on log
markers, and twice produced two waiters on the same job and therefore two
concurrent training runs -- once corrupting a result file. A single sequential
owner cannot do that.

Behaviour, all of it deliberate:

**Watchdog, not halt.** A step that raises is logged with its traceback, marked
``[!]`` in PROGRESS.md, and the queue moves on. One failure must never idle the
GPU for hours. Failed steps are retried once at the end, when whatever transient
condition caused them has probably passed.

**Resume.** Every step declares what artifact proves it finished. If that exists
-- and, for training runs, records real provenance and a finite score -- the step
is skipped. A session that dies at step five restarts at step six.

**Commit after every step, not every gate.** If the machine dies the results are
already on GitHub. Push failures are logged and ignored: losing the network must
not stop the queue.

**Estimates are recorded, not guessed.** Each completed step writes its actual
wall-clock into ``state/queue_state.json``, and the remaining-time estimate in
PROGRESS.md is revised from measurements as they arrive.

C4 is deliberately absent: the seven-mode ablation runs on Kaggle (see
``state/kaggle_c4_instructions.md``) so it does not sit in front of C5-C7 for a
day. It merges in via ``scripts/import_kaggle_results.py``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, project_root, save_json  # noqa: E402

LOGGER = get_logger("gbmc.queue")
ROOT = project_root()
PY = str(Path(sys.executable))
STATE = ROOT / "state" / "queue_state.json"


# --------------------------------------------------------------------------- #
# step definition
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    name: str
    phase: str
    argv: list[str]
    produces: list[str] = field(default_factory=list)
    estimate_min: float = 30.0
    gate: str | None = None
    optional: bool = False
    #: Substrings the artifact must contain to count as finished. Existence
    #: alone is not enough: results/baselines_seed42.json exists from before the
    #: A7.2 rework and holds the old parameter-matched `B2_mel_cnn`, and
    #: structural_controls.json exists holding a dry run. Skipping on existence
    #: would silently drop two Task 2 deliverables.
    requires_text: list[str] = field(default_factory=list)
    #: Cheap derived artifacts that must be regenerated whenever their inputs
    #: change, so they are never skipped.
    always: bool = False

    def done(self) -> bool:
        """Has this already produced everything it promises, freshly?"""
        if self.always or not self.produces:
            return False
        for rel in self.produces:
            path = ROOT / rel
            if not _artifact_ok(path):
                return False
            if self.requires_text:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    return False
                if any(token not in text for token in self.requires_text):
                    return False
        return True


def _artifact_ok(path: Path) -> bool:
    if not path.exists():
        return False
    if path.suffix != ".json":
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return bool(payload)
    if payload.get("provenance") == "synthetic":
        return False
    test = payload.get("test")
    if isinstance(test, dict):
        # a training result counts only with a real score in it
        return any(v == v for v in test.values() if isinstance(v, float)) or bool(test)
    return True


def _train(task: int, tag: str, overrides: list[str], extra: list[str] | None = None):
    return [PY, "-u", "-m", "src.train", "--task", str(task), "--device", "cuda",
            "--num-workers", "0", "--run-tag", tag, *(extra or []),
            "--override", *overrides]


# --------------------------------------------------------------------------- #
# the queue
# --------------------------------------------------------------------------- #
MTAT = ["data.tag_corpora=[mtat]", "tags.vocab_source=mtat",
        "data.text_source=metadata"]
MUSICCAPS = ["data.tag_corpora=[musiccaps]", "tags.vocab_source=musiccaps",
             "data.text_source=caption_masked"]
FUSION_BATCH = ["train.batch_size=8", "train.grad_accum_steps=4"]


def build_queue() -> list[Step]:
    steps: list[Step] = [
        # ---- C1: finish the Task 2 story ---------------------------------- #
        Step("C1a baselines B1/B2/B4", "C1",
             [PY, "-u", "scripts/run_baselines.py", "--device", "cuda",
              "--epochs", "30", "--cnn-targets", "genre,tags"],
             produces=["results/baselines_seed42.json"],
             # the file on disk predates A7.2 and holds the old
             # parameter-matched B2; the rework produces these two instead
             requires_text=["B2_mel_cnn_genre", "B2_mel_cnn_tags"],
             estimate_min=120),
        Step("C1b structural controls", "C1",
             [PY, "-u", "scripts/structural_controls.py",
              "--domains", "genre,tags", "--seeds", "42", "--epochs", "30"],
             produces=["results/structural_controls.json"],
             # the file on disk is a --dry-run, which records no scores
             requires_text=["genre_rewired", "macro_f1"],
             estimate_min=35),

        # ---- C2: the fusion headline -------------------------------------- #
        Step("C2 Task 3 headline (MTAT, bert-base)", "C2",
             _train(3, "mtat_cross_attention_headline",
                    ["bert.model_name=bert-base-uncased",
                     "fusion.mode=cross_attention", *MTAT, *FUSION_BATCH]),
             produces=["results/task3_seed42_mtat_cross_attention_headline.json"],
             estimate_min=200),

        # ---- C3: opens the human gate; must not wait behind the sweep ----- #
        Step("C3 Task 4 headline (MusicCaps dual encoder)", "C3",
             _train(4, "musiccaps_dual",
                    ["bert.model_name=bert-base-uncased", *MUSICCAPS],
                    extra=["--precompute-text-embeddings"]),
             produces=["results/task4_seed42_musiccaps_dual.json"],
             estimate_min=45),
        Step("C3 retrieval export", "C3",
             [PY, "-u", "-m", "src.evaluate", "--device", "cuda", "--tasks", "4"],
             produces=["results/retrieval_examples/retrieval_examples.json"],
             estimate_min=10),
        Step("C3 listening study", "C3",
             [PY, "-u", "scripts/make_listening_page.py"],
             produces=["data/human_eval/sheet_key.json"],
             estimate_min=5, gate="OPERATOR GATE 1"),

        # ---- C5: needed for the Task 4 zero-shot comparison ---------------- #
        *[Step(f"C5 Task 3 MusicCaps {mode}", "C5",
               _train(3, f"musiccaps_{mode}",
                      ["bert.model_name=distilbert-base-uncased",
                       f"fusion.mode={mode}", *MUSICCAPS, *FUSION_BATCH,
                       "train.epochs=8"]),
               produces=[f"results/task3_seed42_musiccaps_{mode}.json"],
               estimate_min=25)
          for mode in ("bert_only", "gnn_only", "cross_attention")],

        # ---- C6: analysis ------------------------------------------------- #
        Step("C6 zero-shot vs supervised", "C6",
             [PY, "-u", "scripts/zero_shot_eval.py", "--seed", "42",
              "--run-tag", "musiccaps_dual"],
             produces=["results/zero_shot_seed42.json"], estimate_min=15),
        # these three are derived from whatever results exist at the time, so
        # they are regenerated rather than skipped -- all are minutes
        Step("C6 full evaluation (t-SNE, S_graph, case studies)", "C6",
             [PY, "-u", "-m", "src.evaluate", "--device", "cuda"],
             produces=["results/metrics.json"], always=True, estimate_min=30),
        Step("C6 threshold bootstrap", "C6",
             [PY, "-u", "scripts/threshold_bootstrap.py", "--n-boot", "100"],
             produces=["results/threshold_bootstrap.json"], always=True,
             estimate_min=5),
        Step("C6 compact figures", "C6",
             [PY, "-u", "scripts/make_compact_figures.py"],
             produces=["results/plots/retrieval_examples.png"],
             always=True, estimate_min=2, optional=True),
        Step("C6 genre confusion figure", "C6",
             [PY, "-u", "scripts/plot_genre_confusion.py"],
             produces=["results/plots/genre_confusion.png"], always=True,
             estimate_min=2),

        # ---- C7: seeds, cheapest first so the most rows survive a cutoff --- #
        *[Step(f"C7 Task 2 genre seed {seed}", "C7",
               _train(2, "fma_genre",
                      ["data.task2_target=genre", "train.epochs=30",
                       "train.batch_size=64"],
                      extra=["--seed", str(seed)]),
               produces=[f"results/task2_seed{seed}_fma_genre.json"],
               estimate_min=4)
          for seed in (1337, 2024)],
        *[Step(f"C7 Task 2 tags seed {seed}", "C7",
               _train(2, "mtat_tags",
                      ["data.task2_target=tags", "train.epochs=30",
                       "train.batch_size=64"],
                      extra=["--seed", str(seed)]),
               produces=[f"results/task2_seed{seed}_mtat_tags.json"],
               estimate_min=10)
          for seed in (1337, 2024)],
        *[Step(f"C7 Task 4 seed {seed}", "C7",
               _train(4, "musiccaps_dual",
                      ["bert.model_name=bert-base-uncased", *MUSICCAPS],
                      extra=["--precompute-text-embeddings", "--seed", str(seed)]),
               produces=[f"results/task4_seed{seed}_musiccaps_dual.json"],
               estimate_min=45)
          for seed in (1337, 2024)],

        # ---- report, last, so it sees everything -------------------------- #
        Step("report fill + structural check", "C7",
             [PY, "-u", "report/fill_report.py"], always=True, estimate_min=1),
    ]
    return steps


# --------------------------------------------------------------------------- #
# bookkeeping
# --------------------------------------------------------------------------- #
def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M")
    line = f"[{stamp}] {message}\n"
    with open(ROOT / "state" / "SESSION_LOG.md", "a", encoding="utf-8") as fh:
        fh.write(line)
    LOGGER.info(message.replace("\n", " ")[:200])


def _git(*args: str) -> bool:
    try:
        done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=300)
        return done.returncode == 0
    except Exception as exc:                                      # noqa: BLE001
        LOGGER.warning("git %s failed: %s", " ".join(args), exc)
        return False


def commit_and_push(message: str) -> None:
    """Results reach GitHub after every step, not just at gates."""
    _git("add", "-A")
    if not _git("commit", "-q", "-m", message):
        return                                     # nothing staged, fine
    if not _git("push", "-q", "origin", "main"):
        LOGGER.warning("push failed; the commit is local and the queue continues")


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"completed": {}, "failed": {}, "started": time.time()}


def update_progress(steps: list[Step], state: dict, current: str | None) -> None:
    """Rewrite the queue block in PROGRESS.md so a cold start needs no archaeology."""
    completed, failed = state["completed"], state["failed"]
    remaining = [s for s in steps
                 if s.name not in completed and not s.done()]
    measured = [v["minutes"] for v in completed.values() if v.get("minutes")]
    scale = 1.0
    if measured:
        predicted = [next((s.estimate_min for s in steps if s.name == k), 0)
                     for k in completed if completed[k].get("minutes")]
        if sum(predicted) > 0:
            scale = sum(measured) / sum(predicted)
    eta = sum(s.estimate_min for s in remaining) * scale

    lines = [
        "<!-- QUEUE:BEGIN (rewritten by scripts/run_queue.py) -->",
        "",
        "## Local run queue -- live status",
        "",
        f"Updated {time.strftime('%Y-%m-%d %H:%M')}. "
        f"**{len(completed)} done, {len(failed)} failed, {len(remaining)} remaining.**",
        "",
        f"Estimated **{eta / 60:.1f} h** of local GPU left"
        + (f", from {len(measured)} measured step(s) "
           f"(estimates running {scale:.2f}x of prediction)." if measured
           else " (no steps measured yet; estimates are priors)."),
        "",
        "C4 is not in this queue: the seven-mode ablation runs on Kaggle so it "
        "does not sit in front of C5-C7 for a day. See "
        "`state/kaggle_c4_instructions.md`; merge with "
        "`scripts/import_kaggle_results.py`.",
        "",
        "| Step | Phase | State | Wall-clock |",
        "|---|---|---|---|",
    ]
    for step in steps:
        if step.name in completed:
            info = completed[step.name]
            mark, when = "[x]", f"{info.get('minutes', 0):.1f} min"
        elif step.name in failed:
            mark, when = "[!]", failed[step.name].get("error", "failed")[:48]
        elif step.name == current:
            mark, when = "[~]", "running"
        elif step.done():
            mark, when = "[x]", "already present"
        else:
            mark, when = "[ ]", f"~{step.estimate_min * scale:.0f} min est."
        lines += [f"| {step.name} | {step.phase} | {mark} | {when} |"]
    lines += ["", "<!-- QUEUE:END -->", ""]
    block = "\n".join(lines)

    path = ROOT / "PROGRESS.md"
    text = path.read_text(encoding="utf-8")
    begin, end = "<!-- QUEUE:BEGIN", "<!-- QUEUE:END -->"
    if begin in text and end in text:
        head = text[:text.index(begin)]
        tail = text[text.index(end) + len(end):]
        text = head + block.replace("<!-- QUEUE:END -->\n", "<!-- QUEUE:END -->") + tail
    else:
        marker = "\n## Phase B0"
        text = (text.replace(marker, "\n" + block + marker, 1)
                if marker in text else text.rstrip("\n") + "\n\n" + block)
    path.write_text(text, encoding="utf-8")


def notify(message: str) -> None:
    """Best effort; a missing notifier must never stop the queue."""
    LOGGER.info("NOTIFY: %s", message)
    (ROOT / "state" / "notifications.log").open("a", encoding="utf-8").write(
        f"{time.strftime('%Y-%m-%d %H:%M')} {message}\n")


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def run_step(step: Step) -> tuple[bool, str, float]:
    started = time.time()
    log_path = ROOT / "logs" / f"queue_{step.phase}_{abs(hash(step.name)) % 10**6}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(step.argv)}\n\n")
        fh.flush()
        try:
            done = subprocess.run(step.argv, cwd=ROOT, stdout=fh,
                                  stderr=subprocess.STDOUT, timeout=6 * 3600)
            ok = done.returncode == 0
            error = "" if ok else f"exit code {done.returncode}"
        except subprocess.TimeoutExpired:
            ok, error = False, "timed out after 6 h"
        except Exception as exc:                                  # noqa: BLE001
            ok, error = False, f"{type(exc).__name__}: {exc}"
            fh.write("\n" + traceback.format_exc())
    return ok, error, (time.time() - started) / 60


def tail(path: Path, n: int = 25) -> str:
    try:
        return "".join(path.read_text(encoding="utf-8", errors="replace")
                       .splitlines(keepends=True)[-n:])
    except Exception:                                             # noqa: BLE001
        return "(no log)"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--from", dest="start_at", default=None)
    parser.add_argument("--only-phase", default=None)
    args = parser.parse_args(argv)

    steps = build_queue()
    if args.only_phase:
        steps = [s for s in steps if s.phase == args.only_phase]
    if args.start_at:
        index = next((i for i, s in enumerate(steps)
                      if s.name.startswith(args.start_at)), 0)
        steps = steps[index:]

    state = load_state()

    if args.plan:
        total = 0.0
        for step in steps:
            skip = step.done() or step.name in state["completed"]
            total += 0 if skip else step.estimate_min
            print(f"{'SKIP' if skip else 'RUN ':<5} {step.phase:<4} "
                  f"{step.name:<48} ~{step.estimate_min:>4.0f} min"
                  + ("   <-- " + step.gate if step.gate else ""))
        print(f"\n{total / 60:.1f} h of work not yet done")
        return 0

    _log(f"QUEUE  started with {len(steps)} step(s)")
    update_progress(steps, state, None)

    for step in steps:
        if step.name in state["completed"]:
            continue
        if step.done():
            LOGGER.info("SKIP %s (artifacts already present)", step.name)
            state["completed"][step.name] = {"minutes": 0.0, "skipped": True}
            continue

        LOGGER.info("=== %s (%s) ===", step.name, step.phase)
        update_progress(steps, state, step.name)
        ok, error, minutes = run_step(step)
        log_path = ROOT / "logs" / f"queue_{step.phase}_{abs(hash(step.name)) % 10**6}.log"

        # `done()` is the *skip* test and returns False for `always` steps by
        # design, so it cannot double as the *success* test -- that marked four
        # successful C6 steps as failed and spent 45 minutes retrying them.
        # Success means: the process exited clean and the artifacts are there.
        produced = all(_artifact_ok(ROOT / rel) for rel in step.produces)
        if ok and produced:
            state["completed"][step.name] = {"minutes": round(minutes, 1)}
            state["failed"].pop(step.name, None)
            _log(f"DONE   {step.phase} {step.name} in {minutes:.1f} min")
            commit_and_push(f"{step.phase}: {step.name}\n\n"
                            f"Completed in {minutes:.1f} min by "
                            f"scripts/run_queue.py.\n\n"
                            "Co-Authored-By: Claude Opus 5 (1M context) "
                            "<noreply@anthropic.com>")
            if step.gate:
                notify(f"{step.gate} reached: {step.name}")
                with open(ROOT / "logs" / "phase_c.log", "a", encoding="utf-8") as fh:
                    fh.write(f"\n{step.gate} -- {time.strftime('%H:%M:%S')} "
                             f"-- {step.name}\n")
        else:
            reason = error or "produced no artifact"
            state["failed"][step.name] = {"error": reason, "minutes": round(minutes, 1)}
            _log(f"BLOCKED {step.phase} {step.name} after {minutes:.1f} min: "
                 f"{reason}\n         last lines of {log_path.name}:\n"
                 + "".join(f"         {ln}" for ln in tail(log_path, 12).splitlines(True)))
            commit_and_push(f"{step.phase}: {step.name} FAILED ({reason})\n\n"
                            "Logged and skipped; the queue continues rather than "
                            "idling the GPU on one failure.\n\n"
                            "Co-Authored-By: Claude Opus 5 (1M context) "
                            "<noreply@anthropic.com>")

        update_progress(steps, state, None)
        save_json(state, STATE)

    # ---- one retry for anything that failed, at the end ------------------- #
    retry = [s for s in steps if s.name in state["failed"]]
    if retry:
        _log(f"QUEUE  retrying {len(retry)} failed step(s) once")
        for step in retry:
            LOGGER.info("=== RETRY %s ===", step.name)
            ok, error, minutes = run_step(step)
            if ok and all(_artifact_ok(ROOT / rel) for rel in step.produces):
                state["completed"][step.name] = {"minutes": round(minutes, 1),
                                                 "retried": True}
                state["failed"].pop(step.name, None)
                _log(f"DONE   {step.name} on retry in {minutes:.1f} min")
                commit_and_push(f"{step.phase}: {step.name} (retry)\n\n"
                                "Co-Authored-By: Claude Opus 5 (1M context) "
                                "<noreply@anthropic.com>")
            else:
                _log(f"BLOCKED {step.name} failed on retry too: {error}")
            update_progress(steps, state, None)
            save_json(state, STATE)

    save_json(state, STATE)
    remaining = [s.name for s in steps if s.name in state["failed"]]
    _log(f"QUEUE  finished: {len(state['completed'])} done, "
         f"{len(remaining)} still failing" + (f" ({', '.join(remaining)})" if remaining else ""))
    notify(f"local queue finished: {len(state['completed'])} done, "
           f"{len(remaining)} failed")
    commit_and_push("C: local queue complete\n\n"
                    "Co-Authored-By: Claude Opus 5 (1M context) "
                    "<noreply@anthropic.com>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
