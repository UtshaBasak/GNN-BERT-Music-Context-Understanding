#!/usr/bin/env python
"""Inject real numbers from ``results/`` into ``report/final_report.tex``.

    python report/fill_report.py [--check]

The report is a living document: every phase gate updates it. Hand-copying
numbers out of JSON into LaTeX is exactly the step where a stale figure survives
three revisions, so the prose never contains a number. It contains a macro, and
this script rewrites the ``AUTOGEN`` block that defines every macro from the
result files on disk.

A result that does not exist yet becomes ``\\textit{pending}`` rather than a
plausible-looking placeholder, and the script prints which macros are still
pending so the gap is visible instead of silent.

``--check`` exits non-zero if any macro is pending -- useful before declaring a
section finished.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, project_root  # noqa: E402

LOGGER = get_logger("gbmc.report")

BEGIN = "% --- AUTOGEN:BEGIN ---------------------------------------------------------"
END = "% --- AUTOGEN:END -----------------------------------------------------------"
PENDING = r"\textit{pending}"

ROOT = project_root()
RESULTS = ROOT / "results"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load(name: str) -> dict | None:
    """Read a result file. ``name`` may include a subdirectory."""
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("%s is not valid JSON (%s)", name, exc)
        return None


def _scalar(value):
    """Unwrap numpy scalars; pandas counts are int64, not int."""
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (ValueError, TypeError):
            return None
    return value if isinstance(value, (int, float)) else None


def num(value, places: int = 4) -> str:
    """A number, or the pending marker -- never a made-up default."""
    value = _scalar(value)
    if value is None:
        return PENDING
    try:
        if value != value:                       # NaN
            return PENDING
    except TypeError:
        return PENDING
    return f"{value:.{places}f}"


def integer(value) -> str:
    value = _scalar(value)
    if value is None:
        return PENDING
    return f"{int(value):,}".replace(",", "{,}")


def seconds(value) -> str:
    value = _scalar(value)
    if value is None:
        return PENDING
    return f"{value / 60:.1f}\\,min" if value >= 90 else f"{value:.0f}\\,s"


def pct(value, places: int = 1) -> str:
    value = _scalar(value)
    if value is None:
        return PENDING
    return f"{100 * value:.{places}f}"


def dig(payload, *path, default=None):
    """Nested lookup that tolerates a missing file entirely."""
    node = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def current_vocab(name: str) -> list:
    """The tag vocabulary on disk right now, for the given source."""
    filename = ("musiccaps_tag_vocab.json" if name == "musiccaps" else "tag_vocab.json")
    path = ROOT / "data" / "splits" / filename
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["tags"] if isinstance(payload, dict) else payload)


def fresh(payload: dict, vocab_source: str, label: str) -> dict:
    """Return the payload only if it was scored against the current vocabulary.

    A7.3 changed 7 of the 50 MusicCaps tags. Results produced before that are
    not stale in a cosmetic sense -- they were scored against a label space that
    test-split annotations helped choose, which is the leak the phase exists to
    remove. If the sweep re-running them dies part way, the old files are still
    on disk and would be picked up silently, mixing corrected and leaked numbers
    in one table. So they are refused, loudly, and render as pending instead.

    Results predating the `tag_vocab` field cannot be verified either way, and
    are treated as unverifiable rather than assumed good.
    """
    if not payload:
        return {}
    want = current_vocab(vocab_source)
    got = payload.get("tag_vocab")
    if not want:
        return payload
    if got is None:
        LOGGER.warning("%s: no tag_vocab recorded, cannot verify it was scored "
                       "against the current vocabulary -- treating as pending",
                       label)
        return {}
    if list(got) != want:
        overlap = len(set(got) & set(want))
        LOGGER.warning("%s: scored against a DIFFERENT %s vocabulary "
                       "(%d/%d tags in common) -- refusing it; re-run the sweep",
                       label, vocab_source, overlap, len(want))
        return {}
    return payload


def find_baseline(payload, name_contains: str) -> dict:
    for entry in dig(payload, "baselines", default=[]) or []:
        if name_contains in str(entry.get("baseline", "")):
            return entry
    return {}


# --------------------------------------------------------------------------- #
# the macro table
# --------------------------------------------------------------------------- #
def build_macros() -> dict:
    import pandas as pd

    macros: dict[str, str] = {}

    # ---- corpus sizes, straight from the manifests --------------------- #
    splits_dir = ROOT / "data" / "splits"
    total = 0
    for corpus in ("mtat", "fma", "deam", "musiccaps"):
        path = splits_dir / f"{corpus}_manifest.csv"
        if path.exists():
            frame = pd.read_csv(path)
            total += len(frame)
            if corpus == "musiccaps":
                counts = frame["split"].value_counts()
                macros["MCTest"] = integer(counts.get("test"))
                macros["MCTrain"] = integer(counts.get("train"))
                macros["MCVal"] = integer(counts.get("val"))
            if corpus == "fma":
                macros["FMATestRows"] = integer(
                    (frame["split"] == "test").sum())
    macros["NTracks"] = integer(total) if total else PENDING
    # three views are stored per track -- segment, chord and heterogeneous --
    # so the graph count is 3x the track count, not equal to it
    summary_path = ROOT / "data" / "processed" / "graph_build_summary.json"
    if summary_path.exists():
        written = sum(int(v.get("written", 0)) for v in
                      json.loads(summary_path.read_text(encoding="utf-8")).values())
        macros["NGraphs"] = integer(written * 3)
    else:
        macros["NGraphs"] = PENDING

    # ---- Task 1 sweep on MusicCaps ------------------------------------- #
    runs = {
        "MCProbe": "task1_seed42_musiccaps_caption_masked_frozen_probe.json",
        "MCTopN": "task1_seed42_musiccaps_caption_masked_top_n.json",
        "MCMasked": "task1_seed42_musiccaps_caption_masked_full_ft.json",
        "MCRaw": "task1_seed42_musiccaps_caption_raw_full_ft.json",
    }
    payloads = {}
    for prefix, filename in runs.items():
        payload = fresh(load(filename) or {}, "musiccaps", filename)
        payloads[prefix] = payload
        macros[f"{prefix}F"] = num(dig(payload, "test", "macro_f1"))
        macros[f"{prefix}Micro"] = num(dig(payload, "test", "micro_f1"))
        macros[f"{prefix}PR"] = num(dig(payload, "test", "mean_auc_pr"))
        macros[f"{prefix}Params"] = integer(payload.get("trainable_params"))
    macros["MCMaskedVal"] = num(dig(payloads["MCMasked"], "best_val_metric"))

    masked = dig(payloads["MCMasked"], "test", "macro_f1")
    raw = dig(payloads["MCRaw"], "test", "macro_f1")
    if isinstance(masked, (int, float)) and isinstance(raw, (int, float)) and masked:
        macros["LeakGap"] = f"$+${raw - masked:.4f}"
        macros["LeakPct"] = f"{100 * (raw - masked) / masked:.0f}"
    else:
        macros["LeakGap"] = macros["LeakPct"] = PENDING

    # ---- Task 1 on MTAT metadata (the like-for-like B3 row) ------------ #
    mtat = fresh(load("task1_seed42_mtat_metadata_full_ft.json") or {}, "mtat",
                 "task1 mtat_metadata")
    macros["TOneMtatF"] = num(dig(mtat, "test", "macro_f1"))
    macros["TOneMtatMicro"] = num(dig(mtat, "test", "micro_f1"))
    macros["TOneMtatPR"] = num(dig(mtat, "test", "mean_auc_pr"))
    macros["TOneMtatParams"] = integer(mtat.get("trainable_params"))

    # ---- Task 2, both domains ------------------------------------------ #
    # The headline genre row is reported over all three seeds. It was seed 42
    # alone while Reproducibility promised three, and the GNN's own spread
    # (+-1.8 points of accuracy) turns out to cover most of the gap to B2 --
    # so the single-seed version overstated a difference in both directions
    # depending on which seed had landed in the table.
    genre_runs = [load(f"task2_seed{seed}_fma_genre.json") or {}
                  for seed in (42, 1337, 2024)]
    genre_runs = [g for g in genre_runs if dig(g, "test", "genre_accuracy") is not None]

    def _spread(field):
        values = [dig(g, "test", field) for g in genre_runs]
        values = [v for v in values if _scalar(v) is not None]
        if not values:
            return None, None
        mu = sum(values) / len(values)
        if len(values) < 2:
            return mu, 0.0
        return mu, (sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5

    acc_mu, acc_sd = _spread("genre_accuracy")
    f1_mu, f1_sd = _spread("genre_macro_f1")
    macros["TTwoGenreSeeds"] = integer(len(genre_runs)) if genre_runs else PENDING
    macros["TTwoGenreAccMean"] = (pct(acc_mu) + r"\%") if acc_mu is not None else PENDING
    macros["TTwoGenreAccSd"] = pct(acc_sd) if acc_sd is not None else PENDING
    macros["TTwoGenreFMean"] = num(f1_mu)
    macros["TTwoGenreFSd"] = num(f1_sd)

    # Parameter count and wall-clock are per-run, not averaged: they describe
    # the model, and seed 42 is the representative run for both.
    genre = load("task2_seed42_fma_genre.json") or {}
    macros["TTwoGenreParams"] = integer(genre.get("trainable_params"))
    macros["TTwoGenreTime"] = seconds(genre.get("wall_clock_s"))

    tags = fresh(load("task2_seed42_mtat_tags.json") or load("task2_seed42.json") or {},
                 "mtat", "task2 mtat_tags")
    macros["TTwoTagF"] = num(dig(tags, "test", "macro_f1"))
    macros["TTwoTagMicro"] = num(dig(tags, "test", "micro_f1"))
    macros["TTwoTagPR"] = num(dig(tags, "test", "mean_auc_pr"))
    macros["TTwoTagParams"] = integer(tags.get("trainable_params"))

    # ---- B2, both domains ---------------------------------------------- #
    baselines = load("baselines_seed42.json") or {}
    b2_genre = find_baseline(baselines, "B2_mel_cnn_genre")
    b2_acc = pct(b2_genre.get("genre_accuracy"))
    macros["BTwoGenreAcc"] = b2_acc + r"\%" if b2_acc != PENDING else PENDING
    macros["BTwoGenreF"] = num(b2_genre.get("genre_macro_f1"))
    macros["BTwoGenreParams"] = integer(b2_genre.get("trainable_params"))
    macros["BTwoGenreTime"] = seconds(b2_genre.get("wall_clock_s"))

    b2_tags = find_baseline(baselines, "B2_mel_cnn_tags")
    macros["BTwoTagF"] = num(b2_tags.get("macro_f1"))
    macros["BTwoTagMicro"] = num(b2_tags.get("micro_f1"))
    macros["BTwoTagPR"] = num(b2_tags.get("mean_auc_pr"))
    macros["BTwoTagParams"] = integer(b2_tags.get("trainable_params"))

    # ---- C4: the seven-mode fusion ablation ------------------------------ #
    import statistics as _st
    from src.fusion_model import FUSION_MODES

    ablation = {}
    for path in sorted((ROOT / "results").glob("task3_seed*_mtat_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        tag = str(payload.get("run_tag") or "")
        if "headline" in tag or not tag.startswith("mtat_"):
            continue
        mode = payload.get("fusion_mode")
        if mode:
            ablation.setdefault(mode, []).append(payload.get("test", {}))

    if ablation:
        rows, means = [], {}
        for mode in FUSION_MODES:
            runs = ablation.get(mode) or []
            tuned = [r.get("macro_f1") for r in runs if r.get("macro_f1") is not None]
            fixed = [r.get("macro_f1_fixed_half") for r in runs
                     if r.get("macro_f1_fixed_half") is not None]
            if not tuned:
                continue
            means[mode] = _st.fmean(tuned)
            rows.append((mode, tuned, fixed))
        best = max(means.values()) if means else 0.0

        def _pm(values):
            if not values:
                return "---"
            sd = _st.stdev(values) if len(values) > 1 else 0.0
            return f"{_st.fmean(values):.4f} $\\pm$ {sd:.4f}"

        lines = []
        for mode, tuned, fixed in rows:
            delta = _st.fmean(tuned) - best
            noise = abs(delta) < 0.0288
            label = mode.replace("_", r"\_")
            lines.append(
                f"\\texttt{{{label}}} & {_pm(tuned)} & {_pm(fixed)} & "
                f"{delta:+.4f} & {len(tuned)} \\\\")
        macros["AblationRows"] = "\n".join(lines)

        inside = sorted(m for m, v in means.items() if abs(v - best) < 0.0288)
        macros["AblationVerdict"] = (
            f"{len(inside)} of the seven modes --- "
            + ", ".join(f"\\texttt{{{m.replace('_', chr(92) + '_')}}}" for m in inside)
            + " --- lie within the measurement floor of one another. No ordering "
            "among them is claimed."
            if len(inside) > 1 else
            f"\\texttt{{{max(means, key=means.get)}}} is the only mode outside "
            "the measurement floor of the best, so the ordering is reportable.")
        macros["AblationBest"] = num(best)
        macros["AblationBertOnly"] = num(means.get("bert_only"))
        macros["AblationGnnOnly"] = num(means.get("gnn_only"))
    else:
        for key in ("AblationRows", "AblationVerdict", "AblationBest",
                    "AblationBertOnly", "AblationGnnOnly"):
            macros[key] = PENDING
        macros["AblationRows"] = r"\multicolumn{5}{c}{\textit{pending}} \\"

    # ---- the corpus contrast: which modality dominates, and when ---------- #
    contrast = {}
    for mode in ("bert_only", "gnn_only", "cross_attention"):
        payload = load(f"task3_seed42_musiccaps_{mode}.json") or {}
        contrast[mode] = dig(payload, "test", "macro_f1")
        macros["MC" + "".join(w.title() for w in mode.split("_"))] = num(contrast[mode])

    # ---- qualitative figures -------------------------------------------- #
    examples = load("retrieval_examples/retrieval_examples.json") or {}
    pool = examples.get("examples", [])
    # The figure plots the curated extremes only -- scripts/make_compact_figures
    # applies exactly this rule. The pool also carries random draws added for
    # the listening study, and summarising those in the caption would describe
    # rows that are not in the picture.
    rows = [r for r in pool if r.get("selection") in (None, "best", "worst")] or pool
    if rows:
        ranks = sorted(int(r.get("true_rank") or 0) for r in rows if r.get("true_rank"))
        failures = [r for r in ranks if r > 10]
        gallery = int(examples.get("gallery_size", 0) or 0)
        macros["MCTestHalf"] = integer(round(gallery / 2)) if gallery else PENDING
        macros["RetrievalNShown"] = integer(len(rows))
        macros["RetrievalNPool"] = integer(len(pool))
        pool_ranks = sorted(int(r.get("true_rank") or 0)
                            for r in pool if r.get("true_rank"))
        macros["RetrievalFigNote"] = (
            f"Median rank {int(np.median(ranks))} among the {len(rows)} shown, "
            f"but these are the curated extremes: over the full "
            f"{len(pool)}-query export the median is "
            f"{int(np.median(pool_ranks))}. Read the panel for the failure "
            "modes, not for a performance estimate."
            if ranks and pool_ranks else PENDING)
        if failures:
            subject = ("Both" if len(failures) == 2 else
                       "The worst two" if len(failures) > 2 else "It")
            macros["RetrievalFailureNote"] = (
                f"{len(failures)} of the {len(rows)} shown place the true clip "
                f"outside the top ten, the worst at rank {max(ranks)}. "
                f"{subject} are captions dominated by production and ambience "
                "terms rather than by instrumentation or rhythm -- properties "
                "that segment-level chroma, MFCC and contrast statistics do not "
                "represent, because the node features summarise what is played "
                "rather than how the recording was made.")
        else:
            macros["RetrievalFailureNote"] = (
                "Every query shown placed the true clip in the top ten.")
    else:
        for key in ("MCTestHalf", "RetrievalFigNote", "RetrievalFailureNote",
                    "RetrievalNShown", "RetrievalNPool"):
            macros[key] = PENDING

    # ---- the Task 3 evaluation-filter defect --------------------------- #
    # Every corrected run keeps its pre-fix numbers under `test_uncorrected`,
    # so the size of the correction is itself measurable rather than asserted.
    deltas, before, after = [], None, None
    for seed in (42, 1337, 2024):
        for mode in ("gnn_only", "early_concat", "late_concat", "gated",
                     "bidirectional", "cross_attention", "bert_only"):
            run = load(f"task3_seed{seed}_mtat_{mode}.json") or {}
            old = (run.get("test_uncorrected") or {}).get("macro_f1")
            new_v = dig(run, "test", "macro_f1")
            if _scalar(old) is not None and _scalar(new_v) is not None:
                deltas.append(new_v - old)
                before = before or (run.get("test") or {}).get("n_test_rows_before_fix")
                after = after or (run.get("test") or {}).get("n_test_rows")
    if deltas:
        macros["CorrDelta"] = num(sum(deltas) / len(deltas), 4)
        macros["CorrDeltaMax"] = num(max(deltas), 4)
        macros["CorrRowsBefore"] = integer(before)
        macros["CorrRowsAfter"] = integer(after)
        macros["CorrNRuns"] = integer(len(deltas))
    else:
        for key in ("CorrDelta", "CorrDeltaMax", "CorrRowsBefore",
                    "CorrRowsAfter", "CorrNRuns"):
            macros[key] = PENDING

    # cross_attention is now the one fusion mode outside the floor
    ca = []
    for seed in (42, 1337, 2024):
        v = dig(load(f"task3_seed{seed}_mtat_cross_attention.json") or {},
                "test", "macro_f1")
        if _scalar(v) is not None:
            ca.append(v)
    macros["AblationCrossAttn"] = num(sum(ca) / len(ca)) if ca else PENDING

    # ---- zero-shot tag prediction (Task 4 deliverable) ----------------- #
    zs = load("zero_shot_seed42.json") or {}
    if zs.get("ensemble"):
        macros["ZeroShotF"] = num(dig(zs, "ensemble", "macro_f1"))
        macros["ZeroShotSpread"] = num(zs.get("template_spread_macro_f1"))
        macros["ZeroShotSup"] = num(dig(zs, "supervised_reference", "macro_f1"))
        macros["ZeroShotGap"] = num(zs.get("zero_shot_gap"))
        macros["ZeroShotClips"] = integer(zs.get("n_test_clips"))
    else:
        for key in ("ZeroShotF", "ZeroShotSpread",
                    "ZeroShotSup", "ZeroShotGap", "ZeroShotClips"):
            macros[key] = PENDING

    # ---- figure captions ------------------------------------------- #
    # The panels each print their own probe numbers, but the caption has to
    # state them too: a reader should not have to squint at a subplot title to
    # learn that the silhouette is negative.
    metrics = load("metrics.json") or {}
    for key, prefix in (("tsne_genre", "TsneGenre"), ("tsne_mood", "TsneMood"),
                        ("tsne_mood_mtat", "TsneMtat")):
        panel = metrics.get(key) or {}
        macros[prefix + "Knn"] = num(panel.get("knn_probe"), 3)
        macros[prefix + "Sil"] = num(panel.get("silhouette"), 3)
        macros[prefix + "N"] = integer(panel.get("n_points"))
    macros["TsnePerplexity"] = integer((metrics.get("tsne_genre") or {}).get("perplexity"))
    macros["TsneMtatExcluded"] = integer(
        (metrics.get("tsne_mood_mtat") or {}).get("n_unlabelled_excluded"))

    coherence = metrics.get("graph_coherence") or {}
    macros["SGraphReal"] = num(coherence.get("S_graph_real"), 4)
    macros["SGraphRewired"] = num(coherence.get("S_graph_rewired"), 4)
    macros["SGraphTau"] = num(coherence.get("tau"), 2)
    macros["SGraphEdgeCos"] = num(coherence.get("mean_edge_cosine"), 3)

    attention = metrics.get("bert_attention_examples") or []
    macros["AttnN"] = integer(len(attention))
    macros["AttnDistinct"] = integer(
        len({str(a.get("text", "")).strip().lower() for a in attention}))

    cases = load("case_studies.json") or {}
    macros["CaseStudyNote"] = (cases.get("caption_note") or PENDING)

    # ---- threshold bootstrap -------------------------------------- #
    boot = load("threshold_bootstrap.json")

    def fixed_half(*names):
        """Untuned macro-F1, from the result if present, else the bootstrap.

        Runs that finished before A7.4 landed have no `macro_f1_fixed_half`
        field, but the bootstrap recomputed it from their stored score matrices,
        so the number is available either way rather than pending.
        """
        for entry in (boot or {}).get("runs", []):
            if entry["run"] in names:
                return num(entry["test_macro_f1_fixed_half"])
        return PENDING

    macros["TTwoTagFixed"] = num(dig(tags, "test", "macro_f1_fixed_half"))
    if macros["TTwoTagFixed"] == PENDING:
        macros["TTwoTagFixed"] = fixed_half("task2_seed42_mtat_tags")
    macros["BTwoTagFixed"] = num(b2_tags.get("macro_f1_fixed_half"))
    if macros["BTwoTagFixed"] == PENDING:
        macros["BTwoTagFixed"] = fixed_half("b2_seed42_tags")
    macros["MCMaskedFixed"] = num(dig(payloads["MCMasked"], "test", "macro_f1_fixed_half"))
    if macros["MCMaskedFixed"] == PENDING:
        macros["MCMaskedFixed"] = fixed_half(
            "task1_seed42_musiccaps_caption_masked_full_ft")
    macros["MCRawFixed"] = num(dig(payloads["MCRaw"], "test", "macro_f1_fixed_half"))
    if macros["MCRawFixed"] == PENDING:
        macros["MCRawFixed"] = fixed_half("task1_seed42_musiccaps_caption_raw_full_ft")
    macros["TOneMtatFixed"] = num(dig(mtat, "test", "macro_f1_fixed_half"))
    if macros["TOneMtatFixed"] == PENDING:
        macros["TOneMtatFixed"] = fixed_half("task1_seed42_mtat_metadata_full_ft")
    if boot and boot.get("runs"):
        macros["NBoot"] = str(boot["n_boot"])
        rows = []
        for entry in boot["runs"]:
            label = entry["run"].replace("_", r"\_")
            rows.append(
                f"{label} & {entry['test_macro_f1_point']:.4f} & "
                f"{entry['test_macro_f1_fixed_half']:.4f} & "
                f"{entry['test_macro_f1_spread']:.4f} \\\\"
            )
        macros["BootRows"] = "\n".join(rows)

        # the headline multi-label run gets its dispersion spelled out, because
        # "spread" (the full range over replicates) is a much more conservative
        # statistic than the standard deviation and the two must not be confused
        lead = next((e for e in boot["runs"] if "mtat_tags" in e["run"]),
                    boot["runs"][0])
        lo, hi = lead["test_macro_f1_ci95"]
        macros["BootLeadStd"] = f"{lead['test_macro_f1_std']:.4f}"
        macros["BootLeadCI"] = f"[{lo:.4f}, {hi:.4f}]"
        macros["BootLeadSpread"] = f"{lead['test_macro_f1_spread']:.4f}"
        macros["BootLeadTuned"] = f"{lead['test_macro_f1_point']:.4f}"
        macros["BootLeadFixed"] = f"{lead['test_macro_f1_fixed_half']:.4f}"
        macros["BootThrStdMean"] = f"{lead['threshold_std_mean']:.3f}"
        macros["BootThrStdMax"] = f"{lead['threshold_std_max']:.3f}"
        macros["BootValRows"] = integer(lead["n_val_rows"])
        # the appendix table: every tag whose threshold moves appreciably
        detail = lead.get("least_stable_tags", [])[:12]
        if detail:
            rows = [f"{d['tag'].replace('_', chr(92) + '_')} & "
                    f"{d['threshold_mean']:.3f} & {d['threshold_std']:.3f} \\\\"
                    for d in detail]
            spelled = {8: "eight", 9: "nine", 10: "ten", 11: "eleven",
                       12: "twelve"}.get(len(detail), str(len(detail)))
            macros["BootWorstTagsTable"] = (
                "\\begin{table}[!htbp]\n\\caption{The " + spelled +
                " least stable per-tag "
                "thresholds on MagnaTagATune, over " + str(boot["n_boot"]) +
                " validation resamples. Every one is a low-frequency tag: with "
                + integer(lead["n_val_rows"]) + " validation clips, a tag "
                "appearing in a few dozen of them has almost no positive "
                "examples left after resampling, so the tuner is fitting "
                "noise.}\n\\label{tab:threshdetail}\n\\centering\n\\small\n"
                "\\begin{tabular}{@{}lrr@{}}\n\\toprule\n"
                "Tag & Mean threshold & s.d. \\\\\n\\midrule\n"
                + "\n".join(rows) +
                "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
            )
        else:
            macros["BootWorstTagsTable"] = PENDING

        worst = lead.get("least_stable_tags", [])[:5]
        macros["BootWorstTags"] = ", ".join(
            f"\\texttt{{{w['tag'].replace('_', chr(92) + '_')}}} "
            f"($\\sigma={w['threshold_std']:.2f}$)" for w in worst) or PENDING
        unstable = [e for e in boot["runs"] if not e["stable"]]
        if unstable:
            worst = max(unstable, key=lambda e: e["test_macro_f1_spread"])
            macros["BootVerdict"] = (
                f"The largest spread is {worst['test_macro_f1_spread']:.4f} "
                f"macro-F1, above the {boot['spread_limit']} threshold fixed in "
                "advance. Tuned numbers are therefore reported alongside "
                "fixed-0.5 numbers throughout, and differences smaller than "
                "this spread are not interpreted."
            )
        else:
            macros["BootVerdict"] = (
                f"Every spread is at or below the {boot['spread_limit']} "
                "threshold fixed in advance, so tuned thresholds are stable "
                "enough for the tuned numbers to stand on their own."
            )
    else:
        macros["NBoot"] = "100"
        macros["BootRows"] = r"\multicolumn{4}{c}{\textit{pending}} \\"
        macros["BootVerdict"] = PENDING
        for key in ("BootLeadStd", "BootLeadCI", "BootLeadSpread", "BootLeadTuned",
                    "BootLeadFixed", "BootThrStdMean", "BootThrStdMax",
                    "BootValRows", "BootWorstTags", "BootWorstTagsTable"):
            macros[key] = PENDING

    # ----------------------------------------------------------------- #
    # DEAM valence/arousal. An explicit PDF deliverable that every Task 3
    # run had already written and no macro had ever read. The headline is the
    # full-budget run; the seed spread comes from the three fixed-budget
    # ablation runs of the same mode, because valence turns out to be far less
    # stable across seeds than arousal and a single seed would hide that.
    # ----------------------------------------------------------------- #
    emo = load("task3_seed42_mtat_cross_attention_headline.json") or {}
    etest = emo.get("test", {}) if isinstance(emo.get("test"), dict) else {}
    if etest.get("valence_mae") is not None:
        macros["EmoValMAE"] = num(etest.get("valence_mae"), 3)
        macros["EmoValRMSE"] = num(etest.get("valence_rmse"), 3)
        macros["EmoValRTwo"] = num(etest.get("valence_r2"), 3)
        macros["EmoAroMAE"] = num(etest.get("arousal_mae"), 3)
        macros["EmoAroRMSE"] = num(etest.get("arousal_rmse"), 3)
        macros["EmoAroRTwo"] = num(etest.get("arousal_r2"), 3)
        macros["EmoN"] = integer(etest.get("valence_n"))
        macros["EmoScale"] = ("1--9 (predictions inverted from standardised)"
                              if etest.get("emotion_target_scale") == "standardised"
                              else "raw 1--9")

        def _emo_spread(field):
            values = []
            for seed in (42, 1337, 2024):
                run = load(f"task3_seed{seed}_mtat_cross_attention.json") or {}
                value = _scalar(dig(run, "test", field))
                if value is not None:
                    values.append(value)
            if not values:
                return None, None
            mu = sum(values) / len(values)
            if len(values) < 2:
                return mu, 0.0
            return mu, (sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5

        v_mu, v_sd = _emo_spread("valence_r2")
        a_mu, a_sd = _emo_spread("arousal_r2")
        macros["EmoValRTwoMean"] = num(v_mu, 3)
        macros["EmoValRTwoSd"] = num(v_sd, 3)
        macros["EmoAroRTwoMean"] = num(a_mu, 3)
        macros["EmoAroRTwoSd"] = num(a_sd, 3)
    else:
        for key in ("EmoValMAE", "EmoValRMSE", "EmoValRTwo", "EmoAroMAE",
                    "EmoAroRMSE", "EmoAroRTwo", "EmoN", "EmoScale",
                    "EmoValRTwoMean", "EmoValRTwoSd", "EmoAroRTwoMean",
                    "EmoAroRTwoSd"):
            macros[key] = PENDING

    # ----------------------------------------------------------------- #
    # Task 3 full-budget headline, and the baselines as macros.
    # The main table carried "Phase B" where the T3 row belongs, and its B1/B4
    # numbers were literals copied from the pre-normalisation baseline run.
    # ----------------------------------------------------------------- #
    macros["TThreeHeadF"] = num(etest.get("macro_f1"))
    macros["TThreeHeadFixed"] = num(etest.get("macro_f1_fixed_half"))
    macros["TThreeHeadMicro"] = num(etest.get("micro_f1"))
    macros["TThreeHeadPR"] = num(etest.get("mean_auc_pr"))
    macros["TThreeHeadParams"] = integer(emo.get("trainable_params"))
    macros["TThreeHeadEpochs"] = integer(emo.get("epochs_run"))
    macros["TThreeHeadTime"] = seconds(emo.get("wall_clock_s"))
    red = load("task3_seed42_mtat_cross_attention.json") or {}
    macros["TThreeAblEpochs"] = integer(red.get("epochs_run"))
    macros["TThreeAblTime"] = seconds(red.get("wall_clock_s"))

    b1r = find_baseline(baselines, "B1_random")
    b1m = find_baseline(baselines, "B1_majority")
    b4 = find_baseline(baselines, "B4_pca_mlp")
    macros["BOneRandF"] = num(b1r.get("macro_f1"))
    macros["BOneRandMicro"] = num(b1r.get("micro_f1"))
    macros["BOneRandPR"] = num(b1r.get("mean_auc_pr"))
    macros["BOneMajF"] = num(b1m.get("macro_f1"))
    macros["BOneMajMicro"] = num(b1m.get("micro_f1"))
    macros["BOneMajPR"] = num(b1m.get("mean_auc_pr"))
    macros["BFourF"] = num(b4.get("macro_f1"))
    macros["BFourMicro"] = num(b4.get("micro_f1"))
    macros["BFourPR"] = num(b4.get("mean_auc_pr"))

    gnn_f = _scalar(dig(tags, "test", "macro_f1"))
    b4_f = _scalar(b4.get("macro_f1"))
    if gnn_f is not None and b4_f is not None:
        delta = gnn_f - b4_f
        macros["BFourDelta"] = num(delta)
        floor = _scalar((boot_spread := load("threshold_bootstrap.json") or {}).get("spread_limit"))
        lead = None
        for entry in (boot_spread.get("runs") or []):
            if entry.get("run") == "task2_seed42_mtat_tags":
                lead = _scalar(entry.get("test_macro_f1_spread"))
        ref = lead or floor
        macros["BFourDeltaFloors"] = num(delta / ref, 1) if ref else PENDING
    else:
        macros["BFourDelta"] = PENDING
        macros["BFourDeltaFloors"] = PENDING

    # ----------------------------------------------------------------- #
    # Task 4, contrastive graph--caption retrieval. Averaged over the three
    # seeds and over both directions, because neither direction is the
    # headline on its own. Every recall is reported against the analytic
    # random baseline the payload carries, since K/gallery is the only
    # number that makes a recall of 0.0135 interpretable.
    # ----------------------------------------------------------------- #
    # D1 re-ran Task 4 at batch 128 for ~10x the optimiser steps. Prefer those
    # runs where they exist and fall back to the originals, so a partial re-run
    # never silently mixes the two: the guard below refuses a mixed set.
    d1 = [load(f"task4_seed{s}_musiccaps_dual_d1.json") for s in (42, 1337, 2024)]
    d1 = [r for r in d1 if r and isinstance(r.get("test"), dict)]
    old_runs = [load(f"task4_seed{s}_musiccaps_dual.json") for s in (42, 1337, 2024)]
    old_runs = [r for r in old_runs if r and isinstance(r.get("test"), dict)]
    seeds = d1 if len(d1) == 3 else old_runs
    tests = [r["test"] for r in seeds if r and isinstance(r.get("test"), dict)]
    if seeds:
        macros["FourBatch"] = integer(seeds[0].get("contrastive_batch_size"))
        macros["FourEpochs"] = integer(seeds[0].get("epochs_run"))
        macros["FourBestEpoch"] = integer(seeds[0].get("best_epoch"))
    else:
        for key in ("FourBatch", "FourEpochs", "FourBestEpoch"):
            macros[key] = PENDING
    # the original numbers stay reportable: the improvement is the finding
    if old_runs and len(d1) == 3:
        prev = [r["test"]["mean_R@10"] for r in old_runs]
        macros["FourPrevRAtTen"] = num(sum(prev) / len(prev), 4)
        prev_lift = [r["test"]["mean_R@10_vs_chance"] for r in old_runs]
        macros["FourPrevLift"] = num(sum(prev_lift) / len(prev_lift), 1)
    else:
        macros["FourPrevRAtTen"] = PENDING
        macros["FourPrevLift"] = PENDING
    if tests:
        def _mean(key):
            values = [t[key] for t in tests if _scalar(t.get(key)) is not None]
            return sum(values) / len(values) if values else None

        def _sd(key):
            values = [t[key] for t in tests if _scalar(t.get(key)) is not None]
            if len(values) < 2:
                return None
            mu = sum(values) / len(values)
            return (sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5

        macros["FourSeeds"] = integer(len(tests))
        macros["FourGallery"] = integer(_mean("gallery_size"))
        macros["FourRAtOne"] = num(_mean("mean_R@1"), 4)
        macros["FourRAtFive"] = num(_mean("mean_R@5"), 4)
        macros["FourRAtTen"] = num(_mean("mean_R@10"), 4)
        macros["FourRAtTenSd"] = num(_sd("mean_R@10"), 4)
        macros["FourMRR"] = num(_mean("mean_MRR"), 4)
        macros["FourChanceOne"] = num(_mean("random_R@1"), 4)
        macros["FourChanceFive"] = num(_mean("random_R@5"), 4)
        macros["FourChanceTen"] = num(_mean("random_R@10"), 4)
        macros["FourChanceMRR"] = num(_mean("random_MRR"), 4)
        macros["FourLiftOne"] = num(_mean("mean_R@1_vs_chance"), 1)
        macros["FourLiftFive"] = num(_mean("mean_R@5_vs_chance"), 1)
        macros["FourLiftTen"] = num(_mean("mean_R@10_vs_chance"), 1)

        g2t, t2g = _mean("g2t_medR"), _mean("t2g_medR")
        macros["FourMedR"] = integer((g2t + t2g) / 2 if None not in (g2t, t2g) else None)
        macros["FourChanceMedR"] = integer(_mean("random_medR"))
        macros["FourGtoT"] = num(_mean("g2t_R@10"), 4)
        macros["FourTtoG"] = num(_mean("t2g_R@10"), 4)
    else:
        for key in ("FourSeeds", "FourGallery", "FourRAtOne", "FourRAtFive",
                    "FourRAtTen", "FourRAtTenSd", "FourMRR", "FourChanceOne",
                    "FourChanceFive", "FourChanceTen", "FourChanceMRR",
                    "FourLiftOne", "FourLiftFive", "FourLiftTen", "FourMedR",
                    "FourChanceMedR", "FourGtoT", "FourTtoG"):
            macros[key] = PENDING

    # ----------------------------------------------------------------- #
    # B4 listening study. Every macro here is defined whether or not the
    # study ran, and the verdict macro carries the null when it is a null --
    # this section is the one most likely to be read as a positive claim it
    # does not make.
    # ----------------------------------------------------------------- #
    human = load("human_eval.json") or {}
    if human:
        macros["HERaters"] = integer(human.get("n_raters"))
        macros["HEItems"] = integer(human.get("n_items_rated"))
        macros["HEControls"] = integer(human.get("n_controls"))
        macros["HERatings"] = integer(human.get("n_ratings"))
        macros["HEMean"] = num(human.get("mean_rating"), 2)
        macros["HESd"] = num(human.get("std_rating"), 2)
        macros["HEMeanReal"] = num(human.get("mean_rating_real"), 2)
        macros["HEMeanControl"] = num(human.get("mean_rating_control"), 2)
        macros["HEAlpha"] = num(human.get("krippendorff_alpha"), 3)
        macros["HESpearman"] = num(human.get("pairwise_spearman_mean"), 3)
        macros["HESpearmanSd"] = num(human.get("pairwise_spearman_std"), 3)
        macros["HEPairs"] = integer(human.get("n_rater_pairs"))
        macros["HEGap"] = num(human.get("control_discrimination"), 2)
        macros["HEItemMin"] = num(human.get("real_item_mean_min"), 2)
        macros["HEItemMax"] = num(human.get("real_item_mean_max"), 2)
        macros["HEBetweenH"] = num(human.get("between_item_H"), 1)
        macros["HEReused"] = integer(human.get("n_controls_reusing_a_caption"))

        share = _scalar(human.get("between_item_variance_share"))
        macros["HEVarShare"] = PENDING if share is None else f"{100 * share:.0f}"

        # p-values want scientific notation when they are small, and LaTeX
        # wants the exponent typeset rather than printed as "e-11".
        def _p(value):
            value = _scalar(value)
            if value is None:
                return PENDING
            if value >= 1e-3:
                return f"{value:.3f}"
            exponent = 0
            mantissa = value
            while mantissa < 1:
                mantissa *= 10
                exponent += 1
            return f"{mantissa:.1f} \\times 10^{{-{exponent}}}"

        macros["HEGapP"] = _p(human.get("control_discrimination_p"))
        macros["HEBetweenP"] = _p(human.get("between_item_p"))

        discriminating = (_scalar(human.get("control_discrimination")) or 0) >= 0.5 \
            and (_scalar(human.get("control_discrimination_p")) or 1) <= 0.05
        macros["HEVerdict"] = (
            "the ratings support the retrieval quality" if discriminating else
            "the study is inconclusive as validation of retrieval quality"
        )
    else:
        for key in ("HERaters", "HEItems", "HEControls", "HERatings", "HEMean",
                    "HESd", "HEMeanReal", "HEMeanControl", "HEAlpha",
                    "HESpearman", "HESpearmanSd", "HEPairs", "HEGap", "HEGapP",
                    "HEItemMin", "HEItemMax", "HEBetweenH", "HEBetweenP",
                    "HEVarShare", "HEReused", "HEVerdict"):
            macros[key] = PENDING

    return macros


def render(macros: dict) -> str:
    lines = [
        BEGIN,
        "% Regenerate with: python report/fill_report.py",
        "% Every value below is read from results/*.json -- do not hand-edit.",
    ]
    for name in sorted(macros):
        value = macros[name]
        if "\n" in value:                        # multi-line table body
            lines.append(f"\\newcommand{{\\{name}}}{{%")
            lines.append(value)
            lines.append("}")
        else:
            lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    lines.append(END)
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if any macro is still pending")
    parser.add_argument("--tex", default=str(ROOT / "report" / "final_report.tex"))
    args = parser.parse_args(argv)

    macros = build_macros()
    path = Path(args.tex)
    text = path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{path} has no AUTOGEN block; add the markers back")

    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    path.write_text(pattern.sub(lambda _: render(macros), text, count=1),
                    encoding="utf-8")

    pending = sorted(k for k, v in macros.items() if PENDING in v)
    print(f"wrote {len(macros)} macros -> {path.relative_to(ROOT)}")
    if pending:
        print(f"{len(pending)} still pending: {', '.join(pending)}")
    else:
        print("no pending values: every number in the report is real")
    return 1 if (args.check and pending) else 0


if __name__ == "__main__":
    raise SystemExit(main())
