"""Manifest construction and the leakage assertion every training run starts with.

Manifest schema (``data/splits/{dataset}_manifest.csv``), frozen by the contract::

    track_id, artist_id, audio_path, text, split, y_genre,
    y_tags (json list), y_valence, y_arousal, duration_s

Two things here are load-bearing:

* **Artist-disjoint splits.** MagnaTagATune is drawn from Magnatune, where one
  artist contributes many clips from the same album. A random clip-level split
  therefore trains and tests on the same recording session and reports a number
  that has nothing to do with generalisation. The standard folder split is
  artist-disjoint by construction, and :func:`assert_no_leakage` enforces it.
* **Synonym merging before top-k selection.** MTAT's raw vocabulary contains
  ``vocal``/``vocals``/``voice`` as separate tags. Counting them separately
  pushes near-duplicates into the top 50 and splits the supervision for one
  concept across three heads.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .utils import (atomic_write_text, ensure_dir, get_logger, load_config,
                    project_root,
                    resolve_path)

LOGGER = get_logger("gbmc.splits")

__all__ = [
    "MANIFEST_COLUMNS",
    "MTAT_SYNONYMS",
    "build_mtat_splits",
    "build_fma_splits",
    "build_deam_splits",
    "build_musiccaps_splits",
    "build_musiccaps_manifest",
    "fma_errata_ids",
    "FMA_SMALL_TRUNCATED",
    "reconcile_across_corpora",
    "build_lmd_inventory",
    "split_summary",
    "reduce_to_top_k_tags",
    "compute_emotion_stats",
    "assert_no_leakage",
    "enforce_artist_disjoint",
    "write_manifest",
    "load_manifest",
    "TEXT_SOURCES",
    "strip_aspect_terms",
    "aspect_surface_forms",
    "mtat_metadata_text",
    "build_text_variants",
    "apply_text_source",
]

MANIFEST_COLUMNS = [
    "track_id", "artist_id", "audio_path", "text", "split",
    "y_genre", "y_tags", "y_valence", "y_arousal", "duration_s",
]

#: canonical -> variants that must be folded into it before counting tag
#: frequencies. Follows the merge list used throughout the MTAT literature.
MTAT_SYNONYMS: dict[str, list[str]] = {
    "beat": ["beats"],
    "chant": ["chanting"],
    "choir": ["choral", "chorus"],
    "classical": ["clasical", "classic"],
    "drum": ["drums"],
    "electronic": ["electro", "electronica", "electric"],
    "fast": ["fast beat", "quick"],
    "female vocal": ["female", "female singer", "female singing", "female vocals",
                     "female voice", "woman", "woman singing", "women"],
    "flute": ["flutes"],
    "guitar": ["guitars"],
    "hard": ["hard rock"],
    "harpsichord": ["harpsicord"],
    "heavy": ["heavy metal", "metal"],
    "horn": ["horns"],
    "india": ["indian"],
    "jazz": ["jazzy"],
    "male vocal": ["male", "male singer", "male singing", "male vocals",
                   "male voice", "man", "man singing", "men"],
    "no beat": ["no drums"],
    "no vocal": ["no singer", "no singing", "no vocals", "no voice", "no voices",
                 "instrumental"],
    "opera": ["operatic"],
    "orchestra": ["orchestral"],
    "quiet": ["silence"],
    "singing": ["singer"],
    "space": ["spacey"],
    "strings": ["string"],
    "synth": ["synthesizer"],
    "violin": ["violins"],
    "vocal": ["vocals", "voice", "voices"],
    "weird": ["strange"],
}

# MTAT's canonical folder split: hex directories 0-b train, c validation, d-f test.
_MTAT_TRAIN_DIRS = set("0123456789ab")
_MTAT_VAL_DIRS = {"c"}
_MTAT_TEST_DIRS = set("def")


# --------------------------------------------------------------------------- #
# manifest io
# --------------------------------------------------------------------------- #
def _portable(path) -> str:
    """A manifest path that survives being cloned onto another machine.

    Manifests are committed, so an absolute path in one publishes the layout of
    the machine that built it and resolves nowhere else. Paths are stored
    relative to the repository root with forward slashes; ``load_audio`` runs
    them back through ``resolve_path``, which leaves an absolute path alone and
    joins a relative one onto the root.
    """
    path = Path(path)
    root = project_root()
    # Deliberately not resolve(): data/raw/<corpus> is typically a junction to
    # the corpus stored outside the repository, and following it would land
    # outside the root and defeat the point.
    absolute = path if path.is_absolute() else (root / path)
    try:
        return absolute.relative_to(root).as_posix()
    except ValueError:
        # genuinely outside the repository: nothing relative to express
        return absolute.as_posix()


def write_manifest(df: pd.DataFrame, path) -> Path:
    """Persist a manifest with exactly the contract columns, in order."""
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = df.copy()
    for column in MANIFEST_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["y_tags"] = frame["y_tags"].apply(
        lambda v: v if isinstance(v, str) else json.dumps(list(v) if _is_seq(v) else [])
    )
    # atomic: a truncated manifest read by the next session is indistinguishable
    # from a real one that simply has fewer rows
    atomic_write_text(out, frame[MANIFEST_COLUMNS].to_csv(index=False))
    LOGGER.info("wrote %d rows -> %s", len(frame), out)
    return out


def load_manifest(path) -> pd.DataFrame:
    """Read a manifest back, parsing ``y_tags`` from JSON into lists."""
    frame = pd.read_csv(resolve_path(path))
    if "y_tags" in frame.columns:
        frame["y_tags"] = frame["y_tags"].apply(_parse_tags)
    return frame


def _parse_tags(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return []


def _is_seq(value) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


# --------------------------------------------------------------------------- #
# tag vocabulary
# --------------------------------------------------------------------------- #
def _synonym_map() -> dict[str, str]:
    mapping = {}
    for canonical, variants in MTAT_SYNONYMS.items():
        mapping[canonical] = canonical
        for variant in variants:
            mapping[variant] = canonical
    return mapping


def reduce_to_top_k_tags(ann: pd.DataFrame, k: int = 50, merge_synonyms: bool = True,
                         id_column: str = "clip_id",
                         drop_columns: Sequence[str] = ("mp3_path",)):
    """Fold synonyms, then keep the ``k`` most frequent tags.

    Returns ``(reduced_frame, tag_names)`` where ``reduced_frame`` has the id
    column plus one binary column per kept tag. Merging happens **before**
    counting: otherwise ``vocal`` (3.9k clips) and ``vocals`` (2.8k) both make
    the cut as separate concepts while a genuinely distinct tag falls out.
    """
    frame = ann.copy()
    ids = frame[id_column].astype(str) if id_column in frame.columns else pd.Series(
        [str(i) for i in range(len(frame))], name=id_column
    )
    tag_columns = [
        c for c in frame.columns
        if c != id_column and c not in set(drop_columns)
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    tags = frame[tag_columns].fillna(0).astype(np.int8)

    if merge_synonyms:
        mapping = _synonym_map()
        merged: dict[str, pd.Series] = {}
        for column in tag_columns:
            canonical = mapping.get(column.strip().lower(), column.strip().lower())
            if canonical in merged:
                merged[canonical] = (merged[canonical] | tags[column].astype(bool))
            else:
                merged[canonical] = tags[column].astype(bool)
        tags = pd.DataFrame({k_: v.astype(np.int8) for k_, v in merged.items()})

    counts = tags.sum(axis=0).sort_values(ascending=False)
    keep = list(counts.index[: int(k)])
    out = pd.concat([ids.reset_index(drop=True), tags[keep].reset_index(drop=True)], axis=1)
    LOGGER.info(
        "tag vocabulary: %d raw -> %d after synonym merge -> top %d kept "
        "(most frequent %s @ %d clips, least %s @ %d)",
        len(tag_columns), tags.shape[1], len(keep),
        keep[0] if keep else "-", int(counts.iloc[0]) if len(counts) else 0,
        keep[-1] if keep else "-", int(counts[keep[-1]]) if keep else 0,
    )
    return out, keep


# --------------------------------------------------------------------------- #
# per-dataset manifests
# --------------------------------------------------------------------------- #
def build_mtat_splits(cfg, validate_audio: bool = False) -> pd.DataFrame:
    """MagnaTagATune manifest using the standard hex-folder split.

    Directories ``0``-``b`` are train, ``c`` is validation, ``d``-``f`` are test.
    That partition is artist-disjoint, which is the only reason MTAT numbers are
    comparable across papers.
    """
    paths = cfg["datasets"]["mtat"]
    ann_path = resolve_path(paths["annotations"])
    info_path = resolve_path(paths["clip_info"])
    audio_root = resolve_path(paths["audio"])

    if not ann_path.exists():
        LOGGER.warning("MTAT annotations not found at %s", ann_path)
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    ann = pd.read_csv(ann_path, sep="\t")
    reduced, tag_names = reduce_to_top_k_tags(
        ann, k=int(cfg["tags"]["top_k"]),
        merge_synonyms=bool(cfg["tags"]["merge_synonyms"]),
    )
    mp3_paths = ann.set_index(ann["clip_id"].astype(str))["mp3_path"].astype(str)

    info = pd.DataFrame()
    if info_path.exists():
        info = pd.read_csv(info_path, sep="\t")
        info["clip_id"] = info["clip_id"].astype(str)
        info = info.set_index("clip_id")

    rows = []
    for _, record in reduced.iterrows():
        clip_id = str(record["clip_id"])
        rel = mp3_paths.get(clip_id, "")
        if not rel or rel == "nan":
            continue
        folder = rel.split("/")[0].lower()
        if folder in _MTAT_TRAIN_DIRS:
            split = "train"
        elif folder in _MTAT_VAL_DIRS:
            split = "val"
        elif folder in _MTAT_TEST_DIRS:
            split = "test"
        else:
            continue

        positive = [t for t in tag_names if int(record[t]) == 1]
        if not positive:
            # clips with no top-50 tag carry no supervision for Task 1
            continue

        artist = title = album = ""
        if len(info) and clip_id in info.index:
            artist = str(info.loc[clip_id, "artist"])
            title = str(info.loc[clip_id, "title"])
            album = str(info.loc[clip_id, "album"]) if "album" in info.columns else ""
        rows.append({
            "track_id": f"mtat_{clip_id}",
            "artist_id": _slug(artist) or f"mtat_unknown_{clip_id}",
            "audio_path": _portable(audio_root / rel),
            # metadata only. Putting the tags here would let Task 1 read
            # its own labels out of Xtext, which is degenerate by construction.
            "text": mtat_metadata_text(title, album, artist),
            "split": split,
            "y_genre": -1,
            "y_tags": json.dumps(positive),
            "y_valence": np.nan,     # MTAT has no emotion labels -> nan sentinel
            "y_arousal": np.nan,
            "duration_s": 29.0,
            "dataset": "mtat",
        })

    frame = pd.DataFrame(rows)
    if validate_audio and len(frame):
        frame = frame[frame["audio_path"].apply(lambda p: Path(p).exists())]
    if len(frame) and bool(cfg.get("splits", {}).get("enforce_artist_disjoint", True)):
        frame = enforce_artist_disjoint(frame)
    if len(frame):
        # MTAT has no captions at all; the only non-circular text is metadata,
        # and it is recorded as every variant so the loader never falls through
        # to a caption that does not exist.
        metadata = dict(zip(frame["track_id"].astype(str), frame["text"].astype(str)))
        variants = build_text_variants(frame, metadata_by_track=metadata)
        variants["caption_raw"] = variants["metadata"]
        variants["caption_masked"] = variants["metadata"]
        atomic_write_text(ensure_dir(cfg["paths"]["splits"]) / "mtat_text_variants.csv",
                          variants.to_csv(index=False))
        empty = int((variants["metadata"].fillna("").str.strip() == "").sum())
        LOGGER.info("MTAT text variants: metadata only; %d/%d clips have no usable "
                    "title/album/artist", empty, len(variants))
    LOGGER.info("MTAT manifest: %d clips %s", len(frame),
                dict(frame["split"].value_counts()) if len(frame) else {})
    return frame


#: The three FMA-small tracks that ship truncated (~1-2 KB instead of ~1 MB) and
#: cannot be decoded. Long-documented in the FMA issue tracker; confirmed on this
#: disk by A1.3. Listed explicitly so the exclusion is auditable even if a future
#: copy of the dataset has them repaired.
FMA_SMALL_TRUNCATED = {99134, 108925, 133297}


def fma_errata_ids(cfg, min_bytes: int = 50_000) -> dict:
    """Track ids to exclude from the FMA manifest, and why.

    Three sources, because no single one is complete:

    * ``not_found.pickle`` shipped with fma_metadata (its ``audio``/``clips``
      lists cover the medium/large subsets, so they usually miss fma_small);
    * :data:`FMA_SMALL_TRUNCATED`, the known-bad small-subset files;
    * a size sweep of what is actually on this disk, which catches a partial
      download that neither list knows about.
    """
    paths = cfg["datasets"]["fma"]
    audio_root = resolve_path(paths["audio"])
    meta_dir = resolve_path(paths["metadata"])
    out: dict = {"not_found_audio": set(), "not_found_clips": set(),
                 "known_truncated": set(FMA_SMALL_TRUNCATED), "undersized": set()}

    pickle_path = meta_dir / "not_found.pickle"
    if pickle_path.exists():
        try:
            import pickle

            with open(pickle_path, "rb") as fh:
                payload = pickle.load(fh)
            out["not_found_audio"] = {int(x) for x in payload.get("audio", [])}
            out["not_found_clips"] = {int(x) for x in payload.get("clips", [])}
        except Exception as exc:  # pragma: no cover - malformed errata
            LOGGER.warning("could not read %s: %s", pickle_path, exc)

    if audio_root.exists():
        for path in audio_root.rglob("*.mp3"):
            try:
                if path.stat().st_size < int(min_bytes):
                    out["undersized"].add(int(path.stem))
            except (OSError, ValueError):
                continue

    out["excluded"] = set().union(*(v for k, v in out.items() if k != "excluded"))
    return out


def build_fma_splits(cfg, validate_audio: bool = False) -> pd.DataFrame:
    """FMA-small manifest using the official ``set.split`` column."""
    paths = cfg["datasets"]["fma"]
    meta_dir = resolve_path(paths["metadata"])
    audio_root = resolve_path(paths["audio"])
    tracks_path = meta_dir / "tracks.csv"
    if not tracks_path.exists():
        LOGGER.warning("FMA tracks.csv not found at %s", tracks_path)
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    subset = tracks[tracks[("set", "subset")] == "small"]
    genres = sorted(subset[("track", "genre_top")].dropna().unique())
    genre_index = {g: i for i, g in enumerate(genres)}

    errata = fma_errata_ids(cfg)
    excluded = errata["excluded"]

    rows, skipped_errata = [], 0
    for track_id, record in subset.iterrows():
        split_raw = str(record[("set", "split")])
        split = {"training": "train", "validation": "val", "test": "test"}.get(split_raw)
        if split is None:
            continue
        if int(track_id) in excluded:
            # a truncated mp3 would decode to silence or throw mid-extraction;
            # either way it is not a data point.
            skipped_errata += 1
            continue
        genre = record[("track", "genre_top")]
        tid = int(track_id)
        rel = f"{tid // 1000:03d}/{tid:06d}.mp3"
        artist = str(record[("artist", "name")])
        title = str(record[("track", "title")])
        rows.append({
            "track_id": f"fma_{tid:06d}",
            "artist_id": _slug(artist) or f"fma_unknown_{tid}",
            "audio_path": _portable(audio_root / rel),
            "text": f"{title} by {artist}. genre: {genre}.",
            "split": split,
            "y_genre": int(genre_index.get(genre, -1)),
            "y_tags": json.dumps([]),     # FMA carries genre, not the tag vocabulary
            "y_valence": np.nan,
            "y_arousal": np.nan,
            "duration_s": 30.0,
            "dataset": "fma",
        })

    frame = pd.DataFrame(rows)
    if validate_audio and len(frame):
        frame = frame[frame["audio_path"].apply(lambda p: Path(p).exists())]
    if len(frame) and bool(cfg.get("splits", {}).get("enforce_artist_disjoint", True)):
        frame = enforce_artist_disjoint(frame)
    frame.attrs["genres"] = genres
    frame.attrs["errata_excluded"] = skipped_errata
    LOGGER.info(
        "FMA errata: excluded %d track(s) — %d known-truncated, %d undersized on "
        "disk, %d from not_found.pickle audio, %d from not_found.pickle clips",
        skipped_errata, len(errata["known_truncated"]), len(errata["undersized"]),
        len(errata["not_found_audio"]), len(errata["not_found_clips"]),
    )
    LOGGER.info("FMA manifest: %d tracks, %d genres %s", len(frame), len(genres),
                dict(frame["split"].value_counts()) if len(frame) else {})
    return frame


def build_deam_splits(cfg, validate_audio: bool = False, seed: int = 42) -> pd.DataFrame:
    """DEAM manifest with artist-disjoint 70/15/15 splits.

    DEAM ships no official split, so we group by artist first and assign whole
    artists to a split. Grouping matters: the 2013/2014 subsets contain several
    excerpts per artist and a random song-level split would leak timbre.
    """
    paths = cfg["datasets"]["deam"]
    ann_dir = resolve_path(paths["annotations"])
    audio_root = resolve_path(paths["audio"])
    meta_dir = resolve_path(paths["metadata"])
    static_dir = ann_dir / "annotations averaged per song" / "song_level"
    if not static_dir.exists():
        LOGGER.warning("DEAM static annotations not found at %s", static_dir)
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    frames = []
    for csv_path in sorted(static_dir.glob("*.csv")):
        part = pd.read_csv(csv_path)
        part.columns = [c.strip() for c in part.columns]
        frames.append(part[["song_id", "valence_mean", "arousal_mean"]])
    annotations = pd.concat(frames, ignore_index=True).drop_duplicates("song_id")

    meta = _load_deam_metadata(meta_dir)
    rows = []
    for record in annotations.itertuples(index=False):
        song_id = int(record.song_id)
        info = meta.get(song_id, {})
        artist = info.get("artist", "")
        title = info.get("title", "")
        genre = info.get("genre", "")
        rows.append({
            "track_id": f"deam_{song_id}",
            "artist_id": _slug(artist) or f"deam_unknown_{song_id}",
            "audio_path": _portable(audio_root / f"{song_id}.mp3"),
            "text": f"{title} by {artist}. genre: {genre}." if title else f"deam excerpt {song_id}",
            "split": "train",     # overwritten by the artist-grouped assignment below
            "y_genre": -1,
            "y_tags": json.dumps([]),      # DEAM has no tag vocabulary -> sentinel
            "y_valence": float(record.valence_mean),
            "y_arousal": float(record.arousal_mean),
            "duration_s": 45.0,
            "dataset": "deam",
        })

    frame = pd.DataFrame(rows)
    if validate_audio and len(frame):
        frame = frame[frame["audio_path"].apply(lambda p: Path(p).exists())]
    if len(frame):
        frame["split"] = _group_split(frame["artist_id"], (0.70, 0.15, 0.15), seed)
    LOGGER.info("DEAM manifest: %d excerpts %s", len(frame),
                dict(frame["split"].value_counts()) if len(frame) else {})
    return frame


def _load_deam_metadata(meta_dir: Path) -> dict[int, dict]:
    """Merge the 2013/2014/2015 metadata files, whose schemas all differ."""
    out: dict[int, dict] = {}
    if not meta_dir.exists():
        return out
    for path in sorted(meta_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:  # pragma: no cover - malformed metadata
            continue
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        id_column = next((c for c in ("song_id", "id") if c in frame.columns), None)
        if id_column is None:
            continue
        for record in frame.to_dict("records"):
            try:
                song_id = int(record[id_column])
            except (TypeError, ValueError):
                continue
            out[song_id] = {
                "artist": str(record.get("artist", "")).strip(),
                "title": str(record.get("song title", record.get("title",
                            record.get("track", "")))).strip(),
                "genre": str(record.get("genre", "")).strip(),
            }
    return out


def build_musiccaps_manifest(cfg, verify_decode: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join the MusicCaps CSV against audio that exists *and decodes*.

    Roughly half of MusicCaps' YouTube sources are gone, and ``yt-dlp`` happily
    leaves behind 0-byte and truncated files that pass ``Path.exists()``. Each
    candidate is therefore opened with soundfile and its duration checked
    against the nominal 10 s clip length.

    The original CSV is never modified. Returns ``(manifest, download_log)``.
    """
    paths = cfg["datasets"]["musiccaps"]
    csv_path = resolve_path(paths["csv"])
    audio_root = resolve_path(paths["audio"])
    nominal = float(paths.get("clip_duration_s", 10.0))
    tolerance = float(paths.get("duration_tol_s", 1.0))

    if not csv_path.exists():
        LOGGER.warning("musiccaps-public.csv not found at %s", csv_path)
        return pd.DataFrame(columns=MANIFEST_COLUMNS), pd.DataFrame()

    source = pd.read_csv(csv_path)
    log_rows, manifest_rows = [], []

    extensions = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus")
    for record in source.itertuples(index=False):
        ytid = str(record.ytid)
        start = int(record.start_s)
        end = int(record.end_s)
        stem = f"{ytid}_{start}_{end}"
        candidates = [audio_root / f"{stem}{ext}" for ext in extensions]
        candidates += [audio_root / f"{ytid}{ext}" for ext in extensions]
        found = next((p for p in candidates if p.exists()), None)

        status, duration = "missing", np.nan
        if found is not None:
            if found.stat().st_size == 0:
                status = "corrupt"
            elif not verify_decode:
                status, duration = "ok", float(end - start)
            else:
                duration = _probe_duration(found)
                if duration is None or not np.isfinite(duration) or duration <= 0:
                    status = "corrupt"
                elif abs(duration - nominal) > tolerance:
                    status = "wrong_duration"
                else:
                    status = "ok"

        log_rows.append({
            "ytid": ytid,
            "start_s": start,
            "end_s": end,
            "is_audioset_eval": bool(record.is_audioset_eval),
            "path": str(found) if found else "",
            "status": status,
            "duration_s": float(duration) if duration is not None and np.isfinite(
                duration if duration is not None else np.nan) else np.nan,
        })
        if status != "ok":
            continue

        aspects = _parse_aspect_list(getattr(record, "aspect_list", ""))
        manifest_rows.append({
            "track_id": f"musiccaps_{stem}",
            # MusicCaps has no artist field; the video id is the finest grouping
            # key available, so each clip is its own "artist" for leakage checks.
            "artist_id": f"yt_{ytid}",
            "audio_path": str(found),
            "text": str(record.caption),
            "split": "test" if bool(record.is_audioset_eval) else "train",
            "y_genre": -1,
            "y_tags": json.dumps(aspects),
            "y_valence": np.nan,
            "y_arousal": np.nan,
            "duration_s": float(duration) if duration is not None else float(end - start),
            "dataset": "musiccaps",
        })

    manifest = pd.DataFrame(manifest_rows)
    log = pd.DataFrame(log_rows)

    if len(log):
        survival = (
            log.assign(ok=log["status"] == "ok")
            .groupby("is_audioset_eval")["ok"].agg(["sum", "count"])
        )
        for is_eval, row in survival.iterrows():
            split_name = "eval (Task 4 gallery)" if is_eval else "train"
            LOGGER.info(
                "MusicCaps survival %s: %d/%d (%.1f%%)",
                split_name, int(row["sum"]), int(row["count"]),
                100.0 * row["sum"] / max(int(row["count"]), 1),
            )
    return manifest, log


def musiccaps_survival_summary(log: pd.DataFrame) -> dict:
    """Per-split survival counts, including the retrieval gallery size."""
    if log is None or not len(log):
        return {}
    out: dict = {"status_counts": log["status"].value_counts().to_dict()}
    for is_eval, group in log.groupby("is_audioset_eval"):
        key = "eval" if is_eval else "train"
        survivors = int((group["status"] == "ok").sum())
        out[f"{key}_nominal"] = int(len(group))
        out[f"{key}_survivors"] = survivors
        out[f"{key}_survival_rate"] = float(survivors / max(len(group), 1))
    out["retrieval_gallery_size"] = out.get("eval_survivors", 0)
    return out


def build_musiccaps_splits(cfg, verify_decode: bool = True) -> pd.DataFrame:
    """MusicCaps manifest; ``is_audioset_eval`` becomes the test split.

    The eval flag is the only principled partition MusicCaps offers, and its
    survivor count *is* the Task 4 retrieval gallery size -- R@10 out of 1,400
    and R@10 out of 2,858 are different claims, so the number is recorded.
    A slice of the non-eval rows is held out for validation.
    """
    manifest, log = build_musiccaps_manifest(cfg, verify_decode=verify_decode)
    splits_dir = ensure_dir(cfg["paths"]["splits"])
    if len(log):
        log.to_csv(splits_dir / "musiccaps_download_log.csv", index=False)
    if not len(manifest):
        return manifest

    train_mask = manifest["split"] == "train"
    train_ids = manifest.loc[train_mask, "track_id"].tolist()
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    n_val = max(1, int(0.1 * len(train_ids)))
    val_ids = set(rng.permutation(train_ids)[:n_val].tolist())
    manifest.loc[manifest["track_id"].isin(val_ids), "split"] = "val"

    # the caption is written *from* the aspect list, so the headline run
    # must not see the labels in its own input. Both variants go to a sidecar
    # and `data.text_source` picks one at load time.
    import ast

    source = pd.read_csv(resolve_path(cfg["datasets"]["musiccaps"]["csv"]))
    aspects_by_track = {}
    for record in source.itertuples(index=False):
        stem = f"{record.ytid}_{int(record.start_s)}_{int(record.end_s)}"
        aspects_by_track[f"musiccaps_{stem}"] = _parse_aspect_list(
            getattr(record, "aspect_list", "")
        )
    variants = build_text_variants(manifest, aspects_by_track=aspects_by_track)
    atomic_write_text(splits_dir / "musiccaps_text_variants.csv",
                      variants.to_csv(index=False))
    stripped = int((variants["caption_raw"] != variants["caption_masked"]).sum())
    LOGGER.info(
        "MusicCaps text variants: %d/%d captions had aspect surface forms removed "
        "(mean %.1f aspects per clip)",
        stripped, len(variants), variants["n_aspects_stripped"].mean(),
    )

    write_manifest(manifest, splits_dir / "musiccaps_manifest.csv")
    return manifest


def _parse_aspect_list(value) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        import ast

        parsed = ast.literal_eval(value)
        return [str(x).strip() for x in parsed] if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def _probe_duration(path: Path) -> float | None:
    """Duration in seconds, or ``None`` if the file cannot be opened."""
    try:
        import soundfile as sf

        with sf.SoundFile(str(path)) as handle:
            if handle.samplerate <= 0:
                return None
            return len(handle) / float(handle.samplerate)
    except Exception:
        pass
    try:  # soundfile cannot open mp3/m4a on every platform; librosa can
        import librosa

        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# split assignment + the leakage assertion
# --------------------------------------------------------------------------- #
def enforce_artist_disjoint(frame: pd.DataFrame, artist_column: str = "artist_id",
                            split_column: str = "split") -> pd.DataFrame:
    """Move whole artists into a single split, keeping their majority split.

    MagnaTagATune's canonical hex-folder split is *not* artist-disjoint in
    practice: Magnatune albums are hashed across directories, so ~57 artists
    have clips in two or three folders. That is a real leak -- the model can win
    by recognising a voice or a room -- and it is the exact failure
    :func:`assert_no_leakage` exists to catch.

    Rather than weaken the assertion, we repair the split: every artist is
    reassigned wholesale to whichever split already holds most of its clips,
    with ties broken train > test > val so the training set absorbs the
    ambiguity. This keeps the folder split's structure (and so stays broadly
    comparable to published MTAT numbers) while making it genuinely
    artist-disjoint. The number of clips moved is logged -- report it.
    """
    if not len(frame) or artist_column not in frame.columns:
        return frame
    out = frame.copy()
    priority = {"train": 0, "test": 1, "val": 2}
    counts = (
        out.groupby([artist_column, split_column]).size().rename("n").reset_index()
    )
    spanning = counts.groupby(artist_column)[split_column].nunique()
    spanning = set(spanning[spanning > 1].index)
    if not spanning:
        # record the zero explicitly: a frame that inherits attrs from a later
        # combined repair would otherwise report that repair's count as its own
        out.attrs["artist_disjoint_moved"] = 0
        out.attrs["artist_disjoint_artists"] = 0
        return out

    counts = counts[counts[artist_column].isin(spanning)].copy()
    counts["_priority"] = counts[split_column].map(lambda s: priority.get(s, 9))
    counts = counts.sort_values(["n", "_priority"], ascending=[False, True])
    winner = counts.drop_duplicates(artist_column).set_index(artist_column)[split_column]

    mask = out[artist_column].isin(spanning)
    before = out.loc[mask, split_column].copy()
    out.loc[mask, split_column] = out.loc[mask, artist_column].map(winner)
    moved = int((out.loc[mask, split_column] != before).sum())
    out.attrs["artist_disjoint_moved"] = moved
    out.attrs["artist_disjoint_artists"] = len(spanning)
    LOGGER.info(
        "artist-disjointness repair: %d artists spanned splits; moved %d of %d "
        "clips (%.2f%% of the corpus) so no artist appears twice",
        len(spanning), moved, len(out), 100.0 * moved / max(len(out), 1),
    )
    return out


def _group_split(groups: pd.Series, ratios: tuple[float, float, float],
                 seed: int = 42) -> pd.Series:
    """Assign whole groups (artists) to train/val/test, greedy by size."""
    counts = groups.value_counts()
    rng = np.random.default_rng(seed)
    order = list(counts.index)
    rng.shuffle(order)

    total = int(counts.sum())
    targets = [ratios[0] * total, ratios[1] * total, ratios[2] * total]
    filled = [0.0, 0.0, 0.0]
    names = ["train", "val", "test"]
    assignment: dict[str, str] = {}
    for group in order:
        deficits = [t - f for t, f in zip(targets, filled)]
        pick = int(np.argmax(deficits))
        assignment[group] = names[pick]
        filled[pick] += counts[group]
    return groups.map(assignment)


def _slug(text: str) -> str:
    text = str(text or "").strip().lower()
    if not text or text == "nan":
        return ""
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def assert_no_leakage(manifest, id_column: str = "track_id",
                      artist_column: str = "artist_id",
                      split_column: str = "split") -> None:
    """Hard-fail if any track id or artist id appears in more than one split.

    Called at the top of every training script. An artist in both train and test
    means the model can win by recognising a voice or a room, and the resulting
    numbers are worthless -- so this raises rather than warns.

    Rows with a blank artist id are treated as their own singleton artist, not
    as one giant "unknown" group that would trigger a false positive.
    """
    if isinstance(manifest, (str, Path)):
        manifest = pd.read_csv(resolve_path(manifest))
    if not isinstance(manifest, pd.DataFrame):
        manifest = pd.DataFrame(manifest)
    if not len(manifest):
        return
    for column in (id_column, split_column):
        if column not in manifest.columns:
            raise KeyError(f"manifest is missing required column {column!r}")

    frame = manifest.copy()
    frame[split_column] = frame[split_column].astype(str)

    dupes = (
        frame.groupby(frame[id_column].astype(str))[split_column]
        .nunique()
        .pipe(lambda s: s[s > 1])
    )
    if len(dupes):
        examples = list(dupes.index[:5])
        raise AssertionError(
            f"LEAKAGE: {len(dupes)} track_id(s) appear in more than one split, "
            f"e.g. {examples}"
        )

    if artist_column in frame.columns:
        artists = frame[artist_column].astype(str).str.strip()
        blank = artists.isin({"", "nan", "none", "unknown"})
        artists = artists.where(~blank, "__solo__" + frame[id_column].astype(str))
        overlap = (
            frame.assign(_artist=artists)
            .groupby("_artist")[split_column]
            .nunique()
            .pipe(lambda s: s[s > 1])
        )
        if len(overlap):
            examples = list(overlap.index[:5])
            raise AssertionError(
                f"LEAKAGE: {len(overlap)} artist_id(s) span multiple splits, "
                f"e.g. {examples}. Splits must be artist-disjoint."
            )
    LOGGER.info(
        "leakage check passed: %d rows, %d tracks, %d artists across %s",
        len(frame), frame[id_column].nunique(),
        frame[artist_column].nunique() if artist_column in frame.columns else -1,
        sorted(frame[split_column].unique()),
    )


def build_all_splits(cfg, validate_audio: bool = False,
                     reconcile: bool = True) -> "dict[str, pd.DataFrame]":
    """Build, reconcile and persist every dataset manifest that has data on disk.

    Three things happen here that do not happen in the individual builders:

    1. **Both MTAT variants** are written. The repaired, artist-disjoint split is
       the headline; the unrepaired canonical folder split is kept under a
       separate name so the literature-comparable number can be cited with its
       caveat.
    2. **Cross-corpus reconciliation.** Every corpus assigns splits on its own,
       so an artist can be leak-free within MTAT and still appear in FMA-train
       and DEAM-test. See :func:`reconcile_across_corpora`.
    3. **The leakage assertion runs on the concatenation**, not just per corpus,
       because that is the object Task 3 and the evaluation actually load.
    """
    splits_dir = ensure_dir(cfg["paths"]["splits"])
    frames: dict = {}

    # --- MTAT, unrepaired: the canonical folder split, for citation only ----- #
    literature_cfg = load_config(cfg.get("_config_path", "config.yaml"),
                                 {"splits.enforce_artist_disjoint": False})
    try:
        mtat_literature = build_mtat_splits(literature_cfg, validate_audio)
    except Exception as exc:
        LOGGER.warning("could not build the unrepaired MTAT variant: %s", exc)
        mtat_literature = pd.DataFrame(columns=MANIFEST_COLUMNS)
    if len(mtat_literature):
        write_manifest(mtat_literature, splits_dir / "mtat_manifest_literature.csv")
        overlap = (
            mtat_literature.groupby("artist_id")["split"].nunique().pipe(lambda x: x[x > 1])
        )
        LOGGER.info(
            "MTAT literature variant persisted UNREPAIRED: %d artists span splits. "
            "Cite it only with that caveat.", len(overlap),
        )

    builders = {
        "mtat": lambda: build_mtat_splits(cfg, validate_audio),
        "fma": lambda: build_fma_splits(cfg, validate_audio),
        "deam": lambda: build_deam_splits(cfg, validate_audio),
        "musiccaps": lambda: build_musiccaps_splits(cfg, verify_decode=validate_audio),
    }
    for name, builder in builders.items():
        try:
            frames[name] = builder()
        except Exception as exc:      # a missing corpus must not stop the others
            LOGGER.warning("could not build %s manifest: %s", name, exc)
            frames[name] = pd.DataFrame(columns=MANIFEST_COLUMNS)

    if reconcile:
        frames = reconcile_across_corpora(frames)

    for name, frame in frames.items():
        if len(frame):
            assert_no_leakage(frame)
            write_manifest(frame, splits_dir / f"{name}_manifest.csv")

    # the object every downstream loader actually sees
    combined = [f for f in frames.values() if len(f)]
    if combined:
        assert_no_leakage(pd.concat(combined, ignore_index=True))
        LOGGER.info("combined leakage check passed across %d corpora", len(combined))

    # the fifth corpus: inventory only, never trained on
    try:
        lmd = build_lmd_inventory(cfg)
        if len(lmd):
            write_manifest(lmd, splits_dir / "lmd_manifest.csv")
        frames["lmd"] = lmd
    except Exception as exc:
        LOGGER.warning("could not inventory Lakh MIDI: %s", exc)

    frames["mtat_literature"] = mtat_literature
    return frames


# --------------------------------------------------------------------------- #
# cross-corpus reconciliation, the LMD inventory, and the summary payload
# --------------------------------------------------------------------------- #
def musiccaps_train_ytids(cfg) -> set:
    """YouTube ids on the MusicCaps **train** split, per the built manifest.

    Manifest track ids are ``musiccaps_<ytid>_<start>_<end>``; the public CSV is
    keyed by bare ``ytid``. Returns an empty set when no manifest exists yet, in
    which case the caller must not silently fall back to counting everything.
    """
    manifest_path = resolve_path(cfg["paths"]["splits"]) / "musiccaps_manifest.csv"
    if not manifest_path.exists():
        return set()
    frame = pd.read_csv(manifest_path)
    train = frame[frame["split"] == "train"]["track_id"].astype(str)
    return {musiccaps_ytid(t) for t in train}


def musiccaps_ytid(track_id: str) -> str:
    """``musiccaps_<ytid>_<start>_<end>`` -> ``<ytid>``.

    Split from the right rather than pattern-matched: YouTube ids themselves
    contain ``-`` and ``_`` (``-0Gj8-vB1q4``), so stripping the known prefix and
    the two trailing integers is the only decomposition that survives them.
    """
    stem = str(track_id)
    if stem.startswith("musiccaps_"):
        stem = stem[len("musiccaps_"):]
    parts = stem.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else stem


def build_musiccaps_tag_vocab(cfg, k: int = 50, split: str = "train") -> tuple[list, dict]:
    """Top-k MusicCaps aspects **counted on the train split only** (A7.3).

    Measured on the real corpus: of MusicCaps' 10.7 aspects per clip, only
    **0.52** appear in the MTAT top-50, and **62% of clips match none of it at
    all**. Training the Task 1 headline against the MTAT vocabulary would
    therefore hand most rows an all-negative label vector -- the model would
    learn to predict nothing and score near-zero macro-F1 for a reason that has
    nothing to do with the model.

    MusicCaps aspects are free text (12,023 distinct strings), so this takes the
    k most frequent after case and whitespace normalisation. Coverage is
    returned alongside so the choice is auditable rather than assumed.

    **Why the split restriction matters.** Choosing *which labels exist* by
    frequency over the whole corpus lets test-split annotations decide the
    vocabulary, which is label information crossing the split boundary before a
    single parameter is trained. It is not hypothetical here: counting over all
    5,521 clips instead of the 2,095 train clips swaps **7 of the 50** tags
    (``e-guitar``, ``fun``, ``keyboard harmony``, ``loud``, ``poor audio
    quality``, ``spirited``, ``youthful`` in, ``calming``, ``classical``,
    ``joyful``, ``keyboard``, ``lively``, ``melodic singing``, ``no other
    instruments`` out). The MTAT vocabulary was checked the same way and is
    unaffected -- its top-50 *set* is identical either way, only the frequency
    ordering moves -- which is why MTAT-scored results did not need re-running
    and the MusicCaps ones did.
    """
    paths = cfg["datasets"]["musiccaps"]
    csv_path = resolve_path(paths["csv"])
    if not csv_path.exists():
        return [], {}

    source = pd.read_csv(csv_path)
    keep_ids = musiccaps_train_ytids(cfg) if split == "train" else None
    if split == "train" and not keep_ids:
        raise RuntimeError(
            "cannot build a train-only MusicCaps vocabulary before the "
            "manifest exists; build splits first, or pass split='all' and say "
            "so in the report"
        )

    counts: dict[str, int] = {}
    per_clip: list[list[str]] = []
    n_counted = 0
    for record in source.itertuples(index=False):
        aspects = [re.sub(r"\s+", " ", str(a).strip().lower())
                   for a in _parse_aspect_list(getattr(record, "aspect_list", ""))]
        aspects = [a for a in aspects if a]
        per_clip.append(aspects)
        if keep_ids is not None and str(getattr(record, "ytid", "")) not in keep_ids:
            continue                      # counted for coverage, not for selection
        n_counted += 1
        for aspect in set(aspects):
            counts[aspect] = counts.get(aspect, 0) + 1

    vocab = [a for a, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: int(k)]]
    chosen = set(vocab)
    hits = [sum(1 for a in aspects if a in chosen) for aspects in per_clip]
    coverage = {
        "split_used": split,
        "n_clips_counted": n_counted,
        "n_clips_in_csv": int(len(source)),
        "n_distinct_aspects": len(counts),
        "vocab_size": len(vocab),
        "mean_aspects_per_clip": float(np.mean([len(a) for a in per_clip])) if per_clip else 0.0,
        "mean_labels_per_clip": float(np.mean(hits)) if hits else 0.0,
        "clips_with_no_label": int(sum(1 for h in hits if h == 0)),
        "clips_with_no_label_pct": float(100.0 * sum(1 for h in hits if h == 0) / max(len(hits), 1)),
        "most_frequent": vocab[:10],
    }
    LOGGER.info(
        "MusicCaps tag vocabulary (%s split, %d clips): %d distinct aspects -> "
        "top %d; mean %.2f labels per clip, %.1f%% of clips left with none",
        split, n_counted, coverage["n_distinct_aspects"], len(vocab),
        coverage["mean_labels_per_clip"], coverage["clips_with_no_label_pct"],
    )
    return vocab, coverage


def compute_emotion_stats(manifest, split: str = "train") -> dict:
    """Mean/std of valence and arousal on the **train** split only (B1.2).

    DEAM's annotations live on a 1-9 scale. Fed straight into an MSE term they
    produce squared errors around 4-10 while per-tag BCE sits near 0.2, so the
    emotion heads absorb almost all the gradient and the tag head never trains.
    ``multitask.auto_balance`` rescales by running magnitude and helps, but it is
    a feedback loop reacting after the fact; standardising the target removes the
    problem at the source and makes the two terms commensurable from step one.

    Train-only, and it refuses any other split, for the same reason the feature
    normalisation statistics do: a target scale fitted on test is test
    information reaching the model.

    Predictions are trained in standardised space and inverted before MAE and
    RMSE are reported, so the numbers in the report stay on the original 1-9
    scale. R2 is invariant to an affine transform of both sides, so it reads the
    same either way.
    """
    if split != "train":
        raise ValueError(
            f"emotion statistics must come from the train split, got {split!r}"
        )
    if isinstance(manifest, (str, Path)):
        manifest = pd.read_csv(resolve_path(manifest))
    rows = manifest[manifest["split"] == "train"]

    stats = {"split": "train"}
    for field in ("y_valence", "y_arousal"):
        values = pd.to_numeric(rows.get(field), errors="coerce").dropna()
        name = field.replace("y_", "")
        if len(values) < 2:
            stats[name] = {"mean": 0.0, "std": 1.0, "n": int(len(values))}
            continue
        std = float(values.std(ddof=0))
        stats[name] = {
            "mean": float(values.mean()),
            "std": std if std > 1e-6 else 1.0,
            "n": int(len(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    LOGGER.info(
        "emotion stats (train only): valence mean %.3f sd %.3f over %d, "
        "arousal mean %.3f sd %.3f over %d",
        stats["valence"]["mean"], stats["valence"]["std"], stats["valence"]["n"],
        stats["arousal"]["mean"], stats["arousal"]["std"], stats["arousal"]["n"],
    )
    return stats


def reconcile_across_corpora(frames: "dict[str, pd.DataFrame]") -> "dict[str, pd.DataFrame]":
    """Make artist ids disjoint across the *concatenation* of all manifests.

    Each corpus assigns its own splits independently, so a manifest can be
    perfectly leak-free on its own while the same artist sits in FMA-train and
    DEAM-test. That is a live problem, not a formality: Task 3 trains on MTAT and
    DEAM jointly and Task 4 evaluates on MusicCaps, so the model can meet in
    training what it is later tested on.

    Whole artists are moved into their majority split, exactly as
    :func:`enforce_artist_disjoint` does within a corpus, and the per-dataset
    move counts are logged.
    """
    present = {name: frame for name, frame in frames.items()
               if frame is not None and len(frame)}
    if len(present) < 2:
        return frames

    combined = pd.concat(
        [frame.assign(_corpus=name) for name, frame in present.items()],
        ignore_index=True,
    )
    before = combined["split"].copy()
    repaired = enforce_artist_disjoint(combined)
    moved = repaired["split"] != before
    if not moved.any():
        LOGGER.info("cross-corpus reconciliation: no artist spans corpora; nothing moved")
        return frames

    per_corpus = repaired.loc[moved].groupby("_corpus").size().to_dict()
    LOGGER.info(
        "cross-corpus reconciliation: moved %d of %d rows so no artist spans two "
        "splits anywhere (%s)",
        int(moved.sum()), len(repaired),
        ", ".join(f"{k}: {v}" for k, v in sorted(per_corpus.items())) or "none",
    )

    out = dict(frames)
    for name in present:
        part = repaired[repaired["_corpus"] == name].drop(columns=["_corpus"])
        # start from the corpus's own attrs, not the combined frame's, so the
        # within-corpus and cross-corpus move counts stay separate numbers
        part.attrs.clear()
        part.attrs.update(present[name].attrs)
        part.attrs["cross_corpus_moved"] = int(per_corpus.get(name, 0))
        out[name] = part.reset_index(drop=True)
    return out


def build_lmd_inventory(cfg) -> pd.DataFrame:
    """Inventory the Lakh MIDI Clean subset.

    Not a training corpus -- it carries no audio and no labels, and is used only
    to bound the chord estimator (see :func:`~src.chords.validate_against_lmd`).
    It gets a manifest anyway so the fifth dataset is auditable: which files were
    present, and which artist each belongs to (the Clean subset encodes
    ``artist/title.mid`` in the path).
    """
    root = resolve_path(cfg["datasets"]["lmd"]["root"])
    if not root.exists():
        LOGGER.warning("Lakh MIDI directory not found: %s", root)
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    rows = []
    for path in sorted(root.rglob("*.mid")) + sorted(root.rglob("*.midi")):
        try:
            artist = path.relative_to(root).parts[0]
        except (ValueError, IndexError):
            artist = "unknown"
        rows.append({
            "track_id": f"lmd_{_slug(artist)}_{_slug(path.stem)}"[:120],
            "artist_id": _slug(artist) or "lmd_unknown",
            "audio_path": str(path),          # a MIDI path, not audio
            "text": f"{path.stem} by {artist}",
            "split": "validation_only",       # never train/val/test: not a training corpus
            "y_genre": -1,
            "y_tags": json.dumps([]),
            "y_valence": np.nan,
            "y_arousal": np.nan,
            "duration_s": np.nan,
            "dataset": "lmd",
        })
    frame = pd.DataFrame(rows)
    # de-duplicate: the Clean subset ships several arrangements per title
    frame = frame.drop_duplicates("track_id").reset_index(drop=True)
    LOGGER.info("LMD inventory: %d MIDI files across %d artists (chord validation only)",
                len(frame), frame["artist_id"].nunique() if len(frame) else 0)
    return frame


def prune_to_cache(cfg, datasets=("mtat", "fma", "deam", "musiccaps")) -> dict:
    """Drop manifest rows that have no cached features, and say how many.

    A row that exists on disk but does not decode survives the manifest build
    (which only checks existence) and then raises a KeyError deep inside a
    dataloader worker. Pruning here means the manifest states what is actually
    usable, which is what every downstream count should be based on.
    """
    import h5py

    processed = resolve_path(cfg["paths"]["processed"])
    splits_dir = resolve_path(cfg["paths"]["splits"])
    report: dict = {}
    for name in datasets:
        manifest_path = splits_dir / f"{name}_manifest.csv"
        h5_path = processed / f"features_{name}.h5"
        if not manifest_path.exists() or not h5_path.exists():
            continue
        frame = pd.read_csv(manifest_path)
        with h5py.File(h5_path, "r") as store:
            cached = set(store.keys())
        keep = frame["track_id"].astype(str).isin(cached)
        dropped = int((~keep).sum())
        if dropped:
            write_manifest(frame[keep], manifest_path)
            LOGGER.info("%s: pruned %d row(s) with no cached features (%d remain)",
                        name, dropped, int(keep.sum()))
        report[name] = {"kept": int(keep.sum()), "dropped": dropped}
    return report


def split_summary(frames: "dict[str, pd.DataFrame]", extra: dict | None = None) -> dict:
    """Per-split clip and artist counts, plus the repair counts, for the report."""
    summary: dict = {"datasets": {}}
    for name, frame in frames.items():
        if frame is None or not len(frame):
            summary["datasets"][name] = {"rows": 0}
            continue
        per_split = {}
        for split, group in frame.groupby("split"):
            per_split[str(split)] = {
                "clips": int(len(group)),
                "artists": int(group["artist_id"].nunique()),
            }
        summary["datasets"][name] = {
            "rows": int(len(frame)),
            "artists": int(frame["artist_id"].nunique()),
            "per_split": per_split,
            "artist_disjoint_moved": int(frame.attrs.get("artist_disjoint_moved", 0)),
            "cross_corpus_moved": int(frame.attrs.get("cross_corpus_moved", 0)),
            "errata_excluded": int(frame.attrs.get("errata_excluded", 0)),
        }
    summary.update(extra or {})
    return summary


# --------------------------------------------------------------------------- #
# Xtext sources, and stripping label surface forms out of captions
# --------------------------------------------------------------------------- #
TEXT_SOURCES = ("caption_masked", "caption_raw", "metadata")

#: Function words that carry no label information and must survive stripping,
#: or the masked caption degenerates into punctuation.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "like", "may",
    "might", "of", "on", "or", "over", "she", "so", "some", "such", "than",
    "that", "the", "their", "them", "there", "these", "they", "this", "those",
    "to", "up", "very", "was", "were", "what", "when", "which", "while", "who",
    "will", "with", "would", "you", "your",
}


def _morphological_variants(word: str) -> set[str]:
    """A word plus the inflections that would defeat exact-string masking.

    Deliberately crude and over-generating rather than linguistically correct:
    a false positive removes one extra word from a caption, while a false
    negative leaves the label visible and inflates the headline number.
    """
    word = word.lower().strip()
    if not word:
        return set()
    forms = {word}

    # strip common suffixes back to a stem
    for suffix in ("ies", "ing", "ers", "ed", "es", "er", "s", "y"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[: -len(suffix)]
            if suffix == "ies":
                stem += "y"
            forms.add(stem)
            if len(stem) > 3 and stem[-1] == stem[-2]:      # "humming" -> "hum"
                forms.add(stem[:-1])

    # ...and generate forwards from every stem we found
    vowels = set("aeiou")
    for stem in list(forms):
        if len(stem) < 3:
            continue
        forms.update({stem + "s", stem + "es", stem + "ing", stem + "ed", stem + "er"})
        if stem.endswith("e"):
            forms.update({stem[:-1] + "ing", stem + "d"})
        if stem.endswith("y"):
            forms.add(stem[:-1] + "ies")
        # consonant doubling: drum -> drummer/drumming, strum -> strumming.
        # Without this the inflected form survives and the label leaks.
        if (len(stem) >= 3 and stem[-1] not in vowels and stem[-2] in vowels
                and stem[-3] not in vowels):
            doubled = stem + stem[-1]
            forms.update({doubled + "ing", doubled + "ed", doubled + "er",
                          doubled + "y"})
    return {f for f in forms if len(f) >= 3}


def aspect_surface_forms(aspects: "Sequence[str]") -> set[str]:
    """Every surface form of an aspect list that could leak the label.

    Returns whole phrases *and* their content tokens with inflections. Phrases
    matter because "sustained strings melody" appears verbatim; tokens matter
    because it also appears scattered, as in "contains sustained strings,
    mellow piano melody".
    """
    forms: set[str] = set()
    for aspect in aspects or []:
        phrase = re.sub(r"\s+", " ", str(aspect).strip().lower())
        if not phrase:
            continue
        forms.add(phrase)
        for token in re.split(r"[^a-z0-9]+", phrase):
            if token and token not in _STOPWORDS and len(token) >= 3:
                forms |= _morphological_variants(token)
    return forms


def strip_aspect_terms(caption: str, aspects: "Sequence[str]",
                       replacement: str = "") -> str:
    """Remove aspect surface forms from a caption before tokenisation.

    MusicCaps captions are written *from* the aspect list, so the labels appear
    almost verbatim in the text. Training Task 1 on the raw caption to predict
    those same aspects measures string matching, not music understanding -- a
    result is not a measure of understanding. This produces the masked variant used for
    the headline number; the raw variant is kept only to quantify the inflation.

    Longest phrases are removed first, so "soft female vocal" is masked as a
    unit rather than being partially consumed by "vocal".
    """
    text = str(caption or "")
    if not text.strip():
        return text
    forms = sorted(aspect_surface_forms(aspects), key=len, reverse=True)
    for form in forms:
        pattern = r"\b" + r"\W+".join(re.escape(p) for p in form.split()) + r"\b"
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # tidy the wreckage: doubled spaces, orphaned punctuation, empty clauses
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:]\s*){2,}", ", ", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"^[\s,;:.]+", "", text)
    return text.strip()


def mtat_metadata_text(title: str = "", album: str = "", artist: str = "") -> str:
    """Non-circular Xtext for MTAT: title + album + artist, never the tags.

    MTAT ships no captions or lyrics. Feeding a track's own tags in as text to
    predict those same tags is degenerate, so the only honest text signal here
    is the `clip_info_final.csv` metadata -- weak, but not circular. The
    weakness is the point of the comparison: fusion gain should scale with text
    informativeness, large on MusicCaps captions and small on this.
    """
    parts = []
    for value in (title, album, artist):
        value = str(value or "").strip()
        if value and value.lower() != "nan":
            parts.append(value)
    return ". ".join(parts) if parts else ""


def build_text_variants(manifest: pd.DataFrame, aspects_by_track: dict | None = None,
                        metadata_by_track: dict | None = None) -> pd.DataFrame:
    """Sidecar table of every Xtext variant, keyed by ``track_id``.

    Kept beside the manifest rather than inside it: the manifest schema is
    frozen at ten columns, and three people are coding against it.
    """
    rows = []
    aspects_by_track = aspects_by_track or {}
    metadata_by_track = metadata_by_track or {}
    for record in manifest.to_dict("records"):
        track_id = str(record["track_id"])
        raw = str(record.get("text", "") or "")
        aspects = aspects_by_track.get(track_id, [])
        rows.append({
            "track_id": track_id,
            "dataset": str(record.get("dataset", "")),
            "caption_raw": raw,
            "caption_masked": strip_aspect_terms(raw, aspects) if aspects else raw,
            "metadata": metadata_by_track.get(track_id, ""),
            "n_aspects_stripped": len(aspects),
        })
    return pd.DataFrame(rows)


def text_variants_path(cfg, dataset: str) -> Path:
    return resolve_path(cfg["paths"]["splits"]) / f"{dataset}_text_variants.csv"


def apply_text_source(manifest: pd.DataFrame, cfg, splits_dir=None) -> pd.DataFrame:
    """Overwrite ``text`` with the variant selected by ``data.text_source``.

    Falls back to whatever is already in ``text`` when a variant is empty for a
    row (some MTAT clips have no usable metadata), so no row silently becomes an
    empty string that tokenises to ``[CLS] [SEP]``.
    """
    source = str(cfg.get("data", {}).get("text_source", "caption_masked"))
    if source not in TEXT_SOURCES:
        raise ValueError(f"data.text_source must be one of {TEXT_SOURCES}, got {source!r}")
    splits_dir = Path(splits_dir) if splits_dir else resolve_path(cfg["paths"]["splits"])

    frame = manifest.copy()
    frame["text_source"] = source
    for dataset in sorted({str(d) for d in frame.get("dataset", pd.Series(dtype=str))}):
        path = splits_dir / f"{dataset}_text_variants.csv"
        if not path.exists():
            continue
        variants = pd.read_csv(path).set_index("track_id")
        if source not in variants.columns:
            continue
        mask = frame["dataset"] == dataset
        chosen = frame.loc[mask, "track_id"].astype(str).map(variants[source])
        current = frame.loc[mask, "text"]
        frame.loc[mask, "text"] = chosen.where(
            chosen.notna() & (chosen.astype(str).str.strip() != ""), current
        )
    LOGGER.info("text_source=%s applied to %d rows", source, len(frame))
    return frame


def main(argv=None) -> int:
    """CLI: build every manifest that has data on disk."""
    import argparse

    from .utils import load_json, load_config, parse_overrides, save_json

    parser = argparse.ArgumentParser(description="Build dataset manifests and splits.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--validate-audio", action="store_true",
                        help="check that every referenced file exists and decodes "
                             "(slow, but the only honest way to size a split)")
    parser.add_argument("--prune-to-cache", action="store_true",
                        help="drop manifest rows with no cached features")
    parser.add_argument("--no-reconcile", action="store_true",
                        help="skip cross-corpus artist reconciliation (not advised)")
    parser.add_argument("--vocab-only", action="store_true",
                        help="re-derive the tag vocabularies from the manifests "
                             "already on disk, without rebuilding any split")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args(argv)

    cfg = load_config(args.config, parse_overrides(args.override))
    if args.vocab_only:
        splits_dir = resolve_path(cfg["paths"]["splits"])
        # the leaky literature variant is kept alongside as a comparison row, so
        # it has to be reloaded too or --vocab-only silently shrinks the summary
        sources = {name: splits_dir / f"{name}_manifest.csv"
                   for name in ("mtat", "fma", "deam", "musiccaps", "lmd")}
        sources["mtat_literature"] = splits_dir / "mtat_manifest_literature.csv"
        frames = {name: pd.read_csv(path) for name, path in sources.items()
                  if path.exists()}
        if not frames:
            raise SystemExit(f"--vocab-only needs manifests in {splits_dir}")
        LOGGER.info("--vocab-only: reusing manifests for %s",
                    ", ".join(sorted(frames)))
    else:
        frames = build_all_splits(cfg, validate_audio=args.validate_audio,
                                  reconcile=not args.no_reconcile)
    pruned = prune_to_cache(cfg) if args.prune_to_cache and not args.vocab_only else {}
    if pruned:
        frames = {name: (pd.read_csv(resolve_path(cfg["paths"]["splits"]) /
                                     f"{name}_manifest.csv")
                         if name in pruned else frame)
                  for name, frame in frames.items()}

    # both vocabularies are selected on the TRAIN split only. Picking
    # which labels exist by frequency over the whole corpus lets test-split
    # annotations decide the label space -- label information crossing the split
    # boundary before any parameter is trained.
    vocab: list[str] = []
    if len(frames.get("mtat", [])):
        ann_path = resolve_path(cfg["datasets"]["mtat"]["annotations"])
        if ann_path.exists():
            ann = pd.read_csv(ann_path, sep="\t")
            ann["clip_id"] = ann["clip_id"].astype(str)
            mtat_frame = frames["mtat"]
            train_ids = set(
                mtat_frame[mtat_frame["split"] == "train"]["track_id"]
                .astype(str).str.removeprefix("mtat_")
            )
            train_ann = ann[ann["clip_id"].isin(train_ids)] if train_ids else ann
            _, vocab = reduce_to_top_k_tags(
                train_ann,
                k=int(cfg["tags"]["top_k"]),
                merge_synonyms=bool(cfg["tags"]["merge_synonyms"]),
            )
            save_json({"tags": vocab, "top_k": int(cfg["tags"]["top_k"]),
                       "merge_synonyms": bool(cfg["tags"]["merge_synonyms"]),
                       "split_used": "train" if train_ids else "all",
                       "n_clips_counted": int(len(train_ann)),
                       "n_clips_in_annotations": int(len(ann))},
                      ensure_dir(cfg["paths"]["splits"]) / "tag_vocab.json")

    # MusicCaps needs its own vocabulary; see build_musiccaps_tag_vocab for why
    mc_vocab, mc_coverage = [], {}
    if len(frames.get("musiccaps", [])):
        try:
            mc_vocab, mc_coverage = build_musiccaps_tag_vocab(
                cfg, k=int(cfg["tags"]["top_k"]), split="train")
            if mc_vocab:
                save_json({"tags": mc_vocab, "top_k": int(cfg["tags"]["top_k"]),
                           "source": "musiccaps_aspect_list",
                           "split_used": mc_coverage.get("split_used", "train"),
                           "coverage": mc_coverage},
                          ensure_dir(cfg["paths"]["splits"]) / "musiccaps_tag_vocab.json")
        except Exception as exc:
            LOGGER.warning("could not build the MusicCaps vocabulary: %s", exc)

    summary = split_summary(frames, extra={
        "tag_vocab_size": len(vocab),
        "musiccaps_tag_vocab_size": len(mc_vocab),
        "musiccaps_vocab_coverage": mc_coverage,
    })

    # MusicCaps survival travels with the split counts: the eval survivor count
    # IS the Task 4 gallery size, and R@K is meaningless without it.
    log_path = resolve_path(cfg["paths"]["splits"]) / "musiccaps_download_log.csv"
    if log_path.exists():
        summary["musiccaps_survival"] = musiccaps_survival_summary(pd.read_csv(log_path))

    save_json(summary, resolve_path(cfg["paths"]["splits"]) / "splits_summary.json")

    # A2 asks for these in results/metrics.json; merge rather than clobber so a
    # later `python -m src.evaluate` does not lose them and vice versa.
    metrics_path = resolve_path(cfg["paths"]["results"]) / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        try:
            metrics = load_json(metrics_path)
        except Exception:
            metrics = {}
    metrics["splits"] = summary
    save_json(metrics, metrics_path)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
