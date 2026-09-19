#!/usr/bin/env python
"""Parse a Google Forms export and report what the ratings actually show.

    python scripts/analyse_human_eval.py

Google Forms exports **wide**: one row per respondent, one column per question,
plus a timestamp and whatever else the form collected. The analysis wants
**long**: ``item_id, rater_id, rating, is_control``. Reshaping that by hand is
error-prone and, worse, silent when it goes wrong -- an off-by-one in the column
order would quietly swap real pairs with controls and invert the headline
finding. So the adapter does it, keyed on ``sheet_key.json``, and refuses rather
than guesses when the columns do not line up.

What gets reported, and why each piece is there:

* **mean +- sd** -- the headline, and on its own almost meaningless;
* **Krippendorff's alpha (ordinal)** -- do raters agree beyond chance? A 1-5
  Likert scale is ordered, so nominal alpha would count 4-vs-5 as badly as
  1-vs-5;
* **control discrimination** -- the gap between real pairs and deliberately
  mismatched ones, with a one-sided test. This is the number that makes the rest
  believable. If raters scored controls as highly as real pairs, the study is
  uninformative and this says so instead of burying it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.human_eval import compute_agreement  # noqa: E402
from src.utils import get_logger, project_root, save_json  # noqa: E402

LOGGER = get_logger("gbmc.humaneval.cli")

#: "Clip 7: how well ..." / "Clip 7" / "7." -- the number is what matters
CLIP_NUMBER = re.compile(r"(?:clip|item|q(?:uestion)?)?\s*#?\s*(\d{1,3})\b", re.I)

#: columns Google Forms adds that are never ratings
NON_RATING = ("timestamp", "email address", "score", "username", "name")


def _rating_columns(frame: pd.DataFrame, expected: int) -> dict[int, str]:
    """Map clip number -> column name, using the number in the question text."""
    found: dict[int, str] = {}
    for column in frame.columns:
        low = str(column).strip().lower()
        if any(low.startswith(skip) for skip in NON_RATING):
            continue
        match = CLIP_NUMBER.search(str(column))
        if not match:
            continue
        number = int(match.group(1))
        if 1 <= number <= expected and number not in found:
            found[number] = column
    return found


def to_long(responses: pd.DataFrame, key: dict) -> pd.DataFrame:
    """Wide Google Forms export -> long ``item_id, rater_id, rating, is_control``."""
    items = {int(row["number"]): row for row in key["items"]}
    expected = len(items)
    columns = _rating_columns(responses, expected)

    if len(columns) < expected:
        missing = sorted(set(items) - set(columns))
        raise SystemExit(
            f"found {len(columns)} rating columns for {expected} clips; could not "
            f"match clip number(s) {missing}. The form questions must contain the "
            "clip number (e.g. 'Clip 7: ...'); regenerate them from "
            "data/human_eval/form_questions.txt rather than renaming by hand."
        )

    lo, hi = key["scale"]["min"], key["scale"]["max"]
    rows = []
    for rater, record in enumerate(responses.to_dict("records"), start=1):
        for number, column in columns.items():
            value = pd.to_numeric(record.get(column), errors="coerce")
            if not np.isfinite(value):
                continue                              # a skipped question
            if not (lo <= value <= hi):
                LOGGER.warning("rater %d gave %s on clip %d, outside %d-%d; dropped",
                               rater, value, number, lo, hi)
                continue
            item = items[number]
            rows.append({
                "item_id": item["item_id"],
                "clip_number": number,
                "rater_id": f"rater_{rater}",
                "rating": float(value),
                "is_control": bool(item["is_control"]),
                "query_track_id": item.get("query_track_id", ""),
                "retrieved_track_id": item.get("retrieved_track_id", ""),
            })
    return pd.DataFrame(rows)


def between_item_discrimination(long: pd.DataFrame) -> dict:
    """Did raters spread the items out, whatever they did with the controls?

    A null control gap has two very different explanations: the panel was not
    listening, or the panel was listening and the controls were not actually
    mismatched. This separates them. If ratings differ across items far beyond
    chance and a large share of the total variance is between items rather than
    within them, the panel was discriminating and the control manipulation is
    what failed.
    """
    from scipy import stats as _stats

    groups = [g["rating"].values for _, g in long.groupby("item_id")]
    if len(groups) < 2:
        return {}
    H, p_value = _stats.kruskal(*groups)

    grand = long["rating"].mean()
    between = sum(len(g) * (g["rating"].mean() - grand) ** 2
                  for _, g in long.groupby("item_id"))
    total = float(((long["rating"] - grand) ** 2).sum())

    means = long.groupby("item_id")["rating"].mean()
    real = long[~long["is_control"]].groupby("item_id")["rating"].mean()
    return {
        "between_item_H": float(H),
        "between_item_p": float(p_value),
        "between_item_variance_share": float(between / total) if total else None,
        "item_mean_min": float(means.min()),
        "item_mean_max": float(means.max()),
        "real_item_mean_min": float(real.min()) if len(real) else None,
        "real_item_mean_max": float(real.max()) if len(real) else None,
    }


def control_caption_reuse(key: dict) -> dict:
    """Do any controls carry a caption that a real item in the sheet also has?

    A control is meant to be a caption paired with audio it does not describe.
    If the same caption is also shown against its true clip elsewhere in the
    sheet, the rater sees one description twice over different audio and has no
    basis for calling either pairing the wrong one -- so the item stops being a
    control and becomes a second opinion on a generic caption.
    """
    real = [it for it in key["items"] if not it["is_control"]]
    reused = []
    for control in (it for it in key["items"] if it["is_control"]):
        twins = [o["item_id"] for o in real if o["caption"] == control["caption"]]
        if twins:
            reused.append({"control_id": control["item_id"],
                           "clip_number": int(control["number"]),
                           "shares_caption_with": twins})
    n_controls = sum(1 for it in key["items"] if it["is_control"])
    return {"n_controls_reusing_a_caption": len(reused),
            "n_controls": n_controls,
            "control_caption_reuse": reused}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", default="data/human_eval/raw_responses.csv")
    parser.add_argument("--key", default="data/human_eval/sheet_key.json")
    parser.add_argument("--out", default="results/human_eval.json")
    args = parser.parse_args(argv)

    root = project_root()
    responses_path, key_path = root / args.responses, root / args.key
    if not key_path.exists():
        raise SystemExit(f"{key_path} not found -- run scripts/make_listening_page.py first")
    if not responses_path.exists():
        raise SystemExit(
            f"{responses_path} not found. Download the Google Form responses as "
            "CSV and save them there; nothing else needs reshaping."
        )

    key = json.loads(key_path.read_text(encoding="utf-8"))
    responses = pd.read_csv(responses_path)
    long = to_long(responses, key)
    if long.empty:
        raise SystemExit("no usable ratings found in the export")

    long_path = root / "data" / "human_eval" / "ratings_long.csv"
    long.to_csv(long_path, index=False)

    n_raters = long["rater_id"].nunique()
    stats = compute_agreement(long.rename(columns={"rater_id": "rater"}))
    stats.update(between_item_discrimination(long))
    stats.update(control_caption_reuse(key))
    stats.update({
        "n_raters": int(n_raters),
        "n_items_rated": int(long["item_id"].nunique()),
        "n_ratings": int(len(long)),
        "n_controls": int(long[long["is_control"]]["item_id"].nunique()),
        "responses_file": str(responses_path.relative_to(root)),
        "meets_five_rater_minimum": bool(n_raters >= 5),
    })

    discrimination = stats.get("control_discrimination")
    p_value = stats.get("control_discrimination_p")
    if discrimination is None:
        stats["verdict"] = ("no control items were rated, so the study cannot "
                            "show that raters were discriminating")
    elif discrimination < 0.5 or (p_value is not None and p_value > 0.05):
        attended = (stats.get("between_item_p") is not None
                    and stats["between_item_p"] < 0.01)
        reused = stats.get("n_controls_reusing_a_caption", 0)
        why = ""
        if attended:
            why = (" Ratings did differ sharply across items "
                   f"(Kruskal-Wallis p = {stats['between_item_p']:.2g}, "
                   f"{stats['between_item_variance_share']:.0%} of variance "
                   "between items), so the panel was attending; the control "
                   "manipulation is what failed, not the panel.")
        if reused:
            why += (f" {reused} of {stats.get('n_controls', 0)} controls carry a "
                    "caption that a real item in the same sheet also carries, "
                    "which makes those controls unidentifiable by construction.")
        stats["verdict"] = (
            f"controls scored within {discrimination:.2f} of real pairs "
            f"(p = {p_value:.3f}). The ratings do NOT establish that raters were "
            "discriminating between true and mismatched pairs, so the human "
            "evaluation is reported as inconclusive rather than as support for "
            "the retrieval quality." + why
        )
    else:
        stats["verdict"] = (
            f"real pairs scored {discrimination:.2f} above controls "
            f"(p = {p_value:.3g}), so raters were discriminating and the mean "
            "rating carries information."
        )

    out = save_json(stats, root / args.out)

    print(f"\n{n_raters} raters, {stats['n_items_rated']} items "
          f"({stats['n_controls']} controls), {stats['n_ratings']} ratings")
    if not stats["meets_five_rater_minimum"]:
        print(f"WARNING: {n_raters} raters is below the 5 the brief asks for")
    print(f"mean rating          {stats.get('mean_rating', float('nan')):.2f} "
          f"+- {stats.get('std_rating', float('nan')):.2f}")
    print(f"  real pairs         {stats.get('mean_rating_real', float('nan')):.2f}")
    print(f"  controls           {stats.get('mean_rating_control', float('nan')):.2f}")
    print(f"Krippendorff alpha   {stats.get('krippendorff_alpha', float('nan')):.3f} (ordinal)")
    print(f"pairwise Spearman    {stats.get('pairwise_spearman_mean', float('nan')):.3f} "
          f"over {stats.get('n_rater_pairs', 0)} pairs")
    print(f"control gap          {stats.get('control_discrimination', float('nan')):+.2f} "
          f"(p = {stats.get('control_discrimination_p', float('nan')):.3g})")
    if stats.get("between_item_H") is not None:
        print(f"between-item         H = {stats['between_item_H']:.1f}, "
              f"p = {stats['between_item_p']:.2g}, "
              f"{stats['between_item_variance_share']:.0%} of variance "
              f"(item means {stats['item_mean_min']:.2f}-{stats['item_mean_max']:.2f})")
    if stats.get("n_controls_reusing_a_caption"):
        print(f"control caption reuse {stats['n_controls_reusing_a_caption']} of "
              f"{stats['n_controls']} controls share a caption with a real item")
    print(f"\n{stats['verdict']}")
    print(f"\nwrote {out} and {long_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
