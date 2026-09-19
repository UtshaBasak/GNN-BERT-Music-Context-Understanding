#!/usr/bin/env python
"""Build the listening study -- sheet key, and a self-contained HTML page.

    python scripts/make_listening_page.py [--n-items 20] [--n-controls 4]

Produces three things in ``data/human_eval/``:

``sheet_key.json``
    The mapping from the numbers a rater sees (1..N) back to item ids, track
    ids and control status. The rater never sees this, and the Google Forms
    adapter needs it to turn a wide export back into long format.

``listening_page.html``
    One self-contained file with the audio embedded as base64, numbered to match
    the form. Self-contained on purpose: a page referencing local file paths
    breaks the moment it is emailed to a listener, and uploading the clips
    somewhere would publish copyrighted audio.

``form_questions.txt``
    The question list to paste into a Google Form, already in the page's order.

**There are deliberately no rating widgets on the page.** Ratings go in the
form, so responses land in one CSV instead of in inboxes, and the page stays a
player rather than a data-collection tool that would need its own storage.

Controls are real audio too -- a caption paired with an unrelated clip. A silent
control would be identifiable without listening, and would measure attention to
silence rather than to musical match.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.human_eval import generate_listening_sheet  # noqa: E402
from src.utils import ensure_dir, get_logger, load_config, project_root, save_json  # noqa: E402

LOGGER = get_logger("gbmc.listening")

MAX_EMBED_BYTES = 2_000_000          # a 10 s mp3 is ~270 KB; this is a sanity cap

PAGE_CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
       max-width: 46rem; margin: 0 auto; padding: 1.5rem; line-height: 1.5; }
h1 { font-size: 1.5rem; margin-bottom: .25rem; }
.lede { color: #555; margin-top: 0; }
ol { padding-left: 0; list-style: none; counter-reset: item; }
li.item { border: 1px solid #d5d5d5; border-radius: 8px; padding: 1rem 1.1rem;
          margin: 1rem 0; }
li.item::before { counter-increment: item; content: "Clip " counter(item);
                  font-weight: 600; display: block; margin-bottom: .5rem; }
.caption { font-style: italic; margin: .5rem 0 .75rem; }
audio { width: 100%; }
.note { background: #f4f4f4; border-left: 3px solid #999; padding: .75rem 1rem;
        margin: 1.25rem 0; border-radius: 4px; }
@media (prefers-color-scheme: dark) {
  li.item { border-color: #444; } .lede { color: #aaa; }
  .note { background: #222; border-left-color: #666; }
}
"""


def _raw_captions(cfg) -> dict:
    """track_id -> the unmasked caption.

    The retrieval examples carry whatever `data.text_source` the model was
    trained on, which for MusicCaps is `caption_masked` -- the label terms are
    stripped out, leaving "The recording features a song that consists,
    alongside, rapping over, and all located in the right channel". That masking
    exists so the *model* cannot read its own labels. A human asked whether a
    description matches a recording needs the actual description, so the study
    reads `caption_raw` from the text variants and refuses to fall back to the
    masked text silently.
    """
    splits = project_root() / cfg["paths"]["splits"]
    out = {}
    for path in sorted(splits.glob("*_text_variants.csv")):
        frame = pd.read_csv(path)
        if "caption_raw" not in frame.columns:
            continue
        for track, caption in zip(frame["track_id"].astype(str),
                                  frame["caption_raw"]):
            if isinstance(caption, str) and caption.strip():
                out[track] = caption.strip()
    return out


def _audio_lookup(cfg) -> dict:
    """track_id -> audio path, across every manifest that has one."""
    splits = project_root() / cfg["paths"]["splits"]
    lookup = {}
    for path in sorted(splits.glob("*_manifest.csv")):
        frame = pd.read_csv(path)
        if "audio_path" not in frame.columns:
            continue
        for track, audio in zip(frame["track_id"].astype(str), frame["audio_path"]):
            if isinstance(audio, str) and audio:
                lookup[track] = audio
    return lookup


def _embed(path: str) -> tuple[str, str] | None:
    """(mime, base64) for a playable file, or None if it cannot be embedded."""
    if not path:
        return None
    file = Path(path)
    if not file.exists():
        LOGGER.warning("audio missing: %s", path)
        return None
    size = file.stat().st_size
    if size > MAX_EMBED_BYTES:
        LOGGER.warning("%s is %.1f MB, too large to embed", file.name, size / 1e6)
        return None
    mime = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
            ".ogg": "audio/ogg", ".opus": "audio/ogg",
            ".flac": "audio/flac"}.get(file.suffix.lower())
    if mime is None:
        LOGGER.warning("%s is not a browser-playable format", file.name)
        return None
    return mime, base64.b64encode(file.read_bytes()).decode("ascii")


def build_page(rows: list[dict], title: str) -> str:
    items = []
    for row in rows:
        caption = html.escape(str(row["caption"]))
        if row.get("audio_b64"):
            player = (f'<audio controls preload="none" '
                      f'src="data:{row["mime"]};base64,{row["audio_b64"]}"></audio>')
        else:
            player = ('<p><strong>Audio unavailable for this clip.</strong> '
                      'Please skip it in the form.</p>')
        items.append(
            f'<li class="item"><div class="caption">&ldquo;{caption}&rdquo;</div>'
            f'{player}</li>'
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{PAGE_CSS}</style></head>
<body>
<h1>{html.escape(title)}</h1>
<p class="lede">Listening study &mdash; {len(rows)} clips, about 10 seconds each.</p>

<div class="note">
<p><strong>What to do.</strong> For each numbered clip below, read the
description, then play the audio. Decide how well the description matches what
you hear, and record your answer in the accompanying form using the
<em>same clip number</em>.</p>
<p><strong>Scale.</strong> 1 = does not match at all &middot; 3 = partly matches
&middot; 5 = matches very well.</p>
<p>Please listen to the whole clip before answering, and work through them in
order. Some descriptions may match poorly &mdash; that is expected and useful,
so please rate what you actually hear rather than what you think the answer
should be.</p>
</div>

<ol>
{chr(10).join(items)}
</ol>
</body></html>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--examples", default="results/retrieval_examples/retrieval_examples.json")
    parser.add_argument("--out-dir", default="data/human_eval")
    parser.add_argument("--n-items", type=int, default=None)
    parser.add_argument("--n-controls", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default="Music description matching study")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    n_items = args.n_items if args.n_items is not None else int(cfg["human_eval"]["n_items"])
    n_controls = (args.n_controls if args.n_controls is not None
                  else int(cfg["human_eval"]["n_controls"]))

    examples_path = project_root() / args.examples
    if not examples_path.exists():
        raise SystemExit(
            f"{examples_path} not found -- run Task 4 and `python -m src.evaluate` "
            "first so there are real retrievals to rate"
        )

    out_dir = ensure_dir(project_root() / args.out_dir)
    sheet = generate_listening_sheet(examples_path, n_items=n_items,
                                     n_controls=n_controls, seed=args.seed,
                                     out_path=out_dir / "listening_sheet.csv")

    audio = _audio_lookup(cfg)
    raw_captions = _raw_captions(cfg)
    if not raw_captions:
        raise SystemExit(
            "no caption_raw found in data/splits/*_text_variants.csv -- the "
            "study must not show raters the masked caption, which has the "
            "descriptive terms stripped out of it"
        )
    rows, missing, unmasked = [], 0, 0
    for position, record in enumerate(sheet.to_dict("records"), start=1):
        track = str(record.get("retrieved_track_id") or "")
        path = record.get("audio_path") or audio.get(track, "")
        embedded = _embed(str(path))
        if embedded is None:
            missing += 1
        # the caption belongs to the QUERY, the audio to what was retrieved
        query = str(record.get("query_track_id") or "")
        caption = raw_captions.get(query)
        if caption:
            unmasked += 1
        else:
            caption = record["caption"]
            LOGGER.warning("no raw caption for %s; showing the stored text", query)
        rows.append({
            "number": position,
            "item_id": record["item_id"],
            "caption": caption,
            "is_control": bool(record["is_control"]),
            "query_track_id": record.get("query_track_id", ""),
            "retrieved_track_id": track,
            "audio_path": str(path),
            "mime": embedded[0] if embedded else None,
            "audio_b64": embedded[1] if embedded else None,
        })

    # ---- the key: numbers back to items. The rater never sees this. -------- #
    key = {
        "seed": args.seed,
        "n_items": n_items,
        "n_controls": n_controls,
        "n_presented": len(rows),
        "scale": {"min": 1, "max": 5,
                  "anchors": {"1": "does not match at all",
                              "3": "partly matches",
                              "5": "matches very well"}},
        "items": [{k: v for k, v in row.items() if k not in ("audio_b64", "mime")}
                  for row in rows],
    }
    save_json(key, out_dir / "sheet_key.json")

    page = build_page(rows, args.title)
    page_path = out_dir / "listening_page.html"
    page_path.write_text(page, encoding="utf-8")

    questions = out_dir / "form_questions.txt"
    with open(questions, "w", encoding="utf-8") as fh:
        fh.write("Paste these into a Google Form as linear-scale (1-5) questions,\n"
                 "in this order, with 'Shuffle question order' switched OFF --\n"
                 "the page is already shuffled and the numbers must line up.\n\n")
        for row in rows:
            fh.write(f"Clip {row['number']}: how well does the description match "
                     f"the audio?\n")

    print(f"wrote {page_path.relative_to(project_root())} "
          f"({page_path.stat().st_size / 1e6:.1f} MB, {len(rows)} clips, "
          f"{sum(r['is_control'] for r in rows)} controls)")
    print(f"captions: {unmasked}/{len(rows)} shown unmasked (caption_raw)")
    if unmasked < len(rows):
        print("WARNING: some clips fall back to the masked caption; raters "
              "cannot judge a description with its descriptive terms removed")
    if len(rows) < n_items + n_controls:
        print(f"WARNING: {len(rows)} clips presented, {n_items + n_controls} "
              "intended -- the retrieval export did not supply enough pairs")
    if missing:
        print(f"WARNING: {missing} clip(s) have no embeddable audio and show a "
              "placeholder; they will be dropped from the analysis")
    print(f"wrote {(out_dir / 'sheet_key.json').relative_to(project_root())}")
    print(f"wrote {questions.relative_to(project_root())}")
    print("\nNext: create the Google Form from form_questions.txt, send it with "
          "the HTML page to >= 5 listeners, then save the responses CSV to "
          "data/human_eval/raw_responses.csv and run "
          "`python scripts/analyse_human_eval.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
