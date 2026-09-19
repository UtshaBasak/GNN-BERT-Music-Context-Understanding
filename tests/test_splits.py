"""The leakage assertion and the tag-vocabulary reduction.

``assert_no_leakage`` runs at the top of every training script, so it has to be
strict in both directions: it must raise on a genuinely leaked manifest, and it
must not raise on a clean one (a false positive would block every run).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.splits import (
    MTAT_SYNONYMS,
    assert_no_leakage,
    load_manifest,
    reduce_to_top_k_tags,
    write_manifest,
)


def _manifest(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


@pytest.fixture
def clean():
    return _manifest([
        {"track_id": "t1", "artist_id": "a1", "split": "train"},
        {"track_id": "t2", "artist_id": "a1", "split": "train"},
        {"track_id": "t3", "artist_id": "a2", "split": "val"},
        {"track_id": "t4", "artist_id": "a3", "split": "test"},
    ])


# --------------------------------------------------------------------------- #
# leakage
# --------------------------------------------------------------------------- #
def test_clean_manifest_passes(clean):
    assert_no_leakage(clean)          # must not raise


def test_track_id_in_two_splits_raises(clean):
    leaked = pd.concat([clean, _manifest([
        {"track_id": "t1", "artist_id": "a1", "split": "test"},
    ])], ignore_index=True)
    with pytest.raises(AssertionError, match="track_id"):
        assert_no_leakage(leaked)


def test_artist_in_two_splits_raises(clean):
    leaked = pd.concat([clean, _manifest([
        {"track_id": "t5", "artist_id": "a1", "split": "test"},
    ])], ignore_index=True)
    with pytest.raises(AssertionError, match="artist_id"):
        assert_no_leakage(leaked)


def test_artist_leak_is_caught_even_with_distinct_track_ids():
    """The subtle case: no repeated track, but the same artist on both sides."""
    leaked = _manifest([
        {"track_id": "x1", "artist_id": "the_band", "split": "train"},
        {"track_id": "x2", "artist_id": "the_band", "split": "test"},
    ])
    with pytest.raises(AssertionError):
        assert_no_leakage(leaked)


def test_blank_artist_ids_are_singletons_not_one_group():
    """Missing artist metadata must not fabricate a leak across every split."""
    frame = _manifest([
        {"track_id": "t1", "artist_id": "", "split": "train"},
        {"track_id": "t2", "artist_id": np.nan, "split": "val"},
        {"track_id": "t3", "artist_id": "unknown", "split": "test"},
    ])
    assert_no_leakage(frame)


def test_empty_manifest_is_not_an_error():
    assert_no_leakage(pd.DataFrame(columns=["track_id", "artist_id", "split"]))


def test_missing_required_column_raises():
    with pytest.raises(KeyError):
        assert_no_leakage(pd.DataFrame({"artist_id": ["a"], "split": ["train"]}))


def test_accepts_a_csv_path(tmp_path, clean):
    path = tmp_path / "m.csv"
    clean.to_csv(path, index=False)
    assert_no_leakage(str(path))


# --------------------------------------------------------------------------- #
# tag vocabulary
# --------------------------------------------------------------------------- #
def _annotations():
    # 'vocal' and 'vocals' are the canonical MTAT duplicate pair
    return pd.DataFrame({
        "clip_id": [1, 2, 3, 4, 5],
        "vocal":   [1, 1, 0, 0, 0],
        "vocals":  [0, 0, 1, 1, 0],
        "guitar":  [1, 0, 1, 0, 1],
        "sitar":   [0, 0, 0, 0, 1],
        "mp3_path": ["a/1.mp3", "b/2.mp3", "c/3.mp3", "d/4.mp3", "e/5.mp3"],
    })


def test_synonyms_merge_before_counting():
    reduced, tags = reduce_to_top_k_tags(_annotations(), k=2, merge_synonyms=True)
    assert "vocals" not in tags, "the variant must be folded into its canonical form"
    assert "vocal" in tags
    # merged 'vocal' now covers 4 clips, so it outranks guitar (3)
    assert int(reduced["vocal"].sum()) == 4
    assert tags[0] == "vocal"


def test_without_merging_the_duplicates_split_the_support():
    reduced, tags = reduce_to_top_k_tags(_annotations(), k=4, merge_synonyms=False)
    assert int(reduced["vocal"].sum()) == 2
    assert int(reduced["vocals"].sum()) == 2


def test_top_k_keeps_exactly_k_tags():
    _, tags = reduce_to_top_k_tags(_annotations(), k=2, merge_synonyms=True)
    assert len(tags) == 2


def test_known_mtat_duplicate_pairs_are_covered():
    """The pairs the spec calls out by name must be in the merge table."""
    flat = {v for variants in MTAT_SYNONYMS.values() for v in variants}
    flat |= set(MTAT_SYNONYMS)
    for pair in ("vocal", "vocals", "choir", "choral", "beat", "beats",
                 "female vocal", "female vocals"):
        assert pair in flat, f"{pair!r} missing from the synonym table"


def test_mp3_path_column_is_not_treated_as_a_tag():
    _, tags = reduce_to_top_k_tags(_annotations(), k=10, merge_synonyms=True)
    assert "mp3_path" not in tags
    assert "clip_id" not in tags


# --------------------------------------------------------------------------- #
# manifest round-trip
# --------------------------------------------------------------------------- #
def test_manifest_round_trip_preserves_tag_lists(tmp_path):
    frame = pd.DataFrame([{
        "track_id": "t1", "artist_id": "a1", "audio_path": "x.mp3",
        "text": "guitar, drum", "split": "train", "y_genre": -1,
        "y_tags": ["guitar", "drum"], "y_valence": np.nan,
        "y_arousal": np.nan, "duration_s": 29.0,
    }])
    path = write_manifest(frame, tmp_path / "m.csv")
    back = load_manifest(path)
    assert back.loc[0, "y_tags"] == ["guitar", "drum"]
    assert np.isnan(back.loc[0, "y_valence"])


def test_written_manifest_has_the_contract_columns(tmp_path):
    frame = pd.DataFrame([{"track_id": "t1", "artist_id": "a1", "split": "train",
                           "y_tags": []}])
    path = write_manifest(frame, tmp_path / "m.csv")
    columns = list(pd.read_csv(path).columns)
    assert columns == ["track_id", "artist_id", "audio_path", "text", "split",
                       "y_genre", "y_tags", "y_valence", "y_arousal", "duration_s"]


# --------------------------------------------------------------------------- #
# the tag vocabularies must be selected on the train split only
#
# Choosing *which labels exist* by frequency over the whole corpus lets
# test-split annotations decide the label space. That is label information
# crossing the split boundary before a single parameter is trained, and it is
# not hypothetical: counting MusicCaps aspects over all 5,521 clips instead of
# the 2,095 train clips swaps 7 of the 50 tags.
# --------------------------------------------------------------------------- #
def _vocab_file(name):
    from src.utils import project_root

    path = project_root() / "data" / "splits" / name
    if not path.exists():
        pytest.skip(f"{name} not built yet")
    return json.loads(path.read_text(encoding="utf-8"))


def test_musiccaps_vocab_declares_train_provenance():
    payload = _vocab_file("musiccaps_tag_vocab.json")
    assert payload.get("split_used") == "train", (
        "musiccaps_tag_vocab.json does not record that it was built on the "
        "train split; rebuild with `python -m src.splits --vocab-only`"
    )
    coverage = payload.get("coverage", {})
    assert coverage.get("n_clips_counted", 0) < coverage.get("n_clips_in_csv", 0), (
        "the vocabulary counted every clip in the CSV, so the test split "
        "helped choose the label space"
    )


def test_mtat_vocab_declares_train_provenance():
    payload = _vocab_file("tag_vocab.json")
    assert payload.get("split_used") == "train"
    assert payload.get("n_clips_counted", 0) < payload.get("n_clips_in_annotations", 0)


def test_musiccaps_vocab_matches_a_train_only_recount():
    """Rebuild the vocabulary from the raw CSV and require an exact match."""
    from src.splits import build_musiccaps_tag_vocab, musiccaps_train_ytids
    from src.utils import load_config, project_root

    saved = _vocab_file("musiccaps_tag_vocab.json")
    cfg = load_config(project_root() / "config.yaml")
    if not musiccaps_train_ytids(cfg):
        pytest.skip("MusicCaps manifest not built yet")
    if not (project_root() / cfg["datasets"]["musiccaps"]["csv"]).exists():
        pytest.skip("MusicCaps CSV not present (it is gitignored raw data)")

    rebuilt, coverage = build_musiccaps_tag_vocab(cfg, k=len(saved["tags"]),
                                                  split="train")
    assert rebuilt == saved["tags"], (
        "the stored MusicCaps vocabulary is not what a train-only recount "
        "produces -- it is stale or was built on the wrong split"
    )
    assert coverage["split_used"] == "train"


def test_counting_every_split_would_change_the_musiccaps_vocabulary():
    """The guard above is only meaningful if the leak would actually show.

    If train-only and all-split selection happened to agree, the provenance
    tests would pass whether or not the fix were in place. They do not agree
    here -- 7 of 50 tags differ -- so this pins the fact that the restriction
    is load-bearing rather than cosmetic.
    """
    from src.splits import build_musiccaps_tag_vocab, musiccaps_train_ytids
    from src.utils import load_config, project_root

    cfg = load_config(project_root() / "config.yaml")
    if not musiccaps_train_ytids(cfg):
        pytest.skip("MusicCaps manifest not built yet")
    if not (project_root() / cfg["datasets"]["musiccaps"]["csv"]).exists():
        pytest.skip("MusicCaps CSV not present (it is gitignored raw data)")

    train_only, _ = build_musiccaps_tag_vocab(cfg, k=50, split="train")
    all_splits, _ = build_musiccaps_tag_vocab(cfg, k=50, split="all")
    assert set(train_only) != set(all_splits), (
        "train-only and all-split selection now agree, so the provenance "
        "tests can no longer detect the leak -- re-derive the guard"
    )


def test_musiccaps_train_vocab_refuses_to_guess_without_a_manifest(tmp_path):
    """No manifest must be an error, not a silent fall back to all splits."""
    from src.splits import build_musiccaps_tag_vocab
    from src.utils import load_config, project_root

    cfg = load_config(project_root() / "config.yaml")
    csv_path = project_root() / cfg["datasets"]["musiccaps"]["csv"]
    if not csv_path.exists():
        pytest.skip("MusicCaps CSV not present")
    cfg["paths"]["splits"] = str(tmp_path)          # no manifest lives here
    with pytest.raises(RuntimeError, match="train-only"):
        build_musiccaps_tag_vocab(cfg, k=50, split="train")


def test_musiccaps_ytid_survives_ids_containing_dashes_and_underscores():
    from src.splits import musiccaps_ytid

    assert musiccaps_ytid("musiccaps_-0Gj8-vB1q4_30_40") == "-0Gj8-vB1q4"
    assert musiccaps_ytid("musiccaps_a_b_c_10_20") == "a_b_c"
    assert musiccaps_ytid("-0Gj8-vB1q4_30_40") == "-0Gj8-vB1q4"


# --------------------------------------------------------------------------- #
# committed manifests must work on a machine that is not the one that built them
# --------------------------------------------------------------------------- #
def test_committed_manifests_hold_no_absolute_paths():
    """An absolute path in a committed manifest resolves nowhere else.

    The manifests are the only preprocessing this repository ships. Storing
    `B:\...` in them makes that shipment useless to anyone who clones it, and
    publishes the directory layout of the machine that built it. `load_audio`
    runs every path through `resolve_path`, which joins a relative path onto
    the repository root, so the relative form is what belongs on disk.
    """
    import csv

    from src.utils import project_root

    splits = project_root() / "data" / "splits"
    offenders = {}
    for path in sorted(splits.glob("*_manifest.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "audio_path" not in (reader.fieldnames or []):
                continue
            bad = [row["audio_path"] for row in reader
                   if row["audio_path"] and (
                       row["audio_path"][1:3] == ":\\"
                       or row["audio_path"].startswith("/"))]
        if bad:
            offenders[path.name] = (len(bad), bad[0])

    assert not offenders, (
        "absolute audio paths in committed manifests: "
        + "; ".join(f"{n} in {f} (e.g. {ex})" for f, (n, ex) in offenders.items())
    )
