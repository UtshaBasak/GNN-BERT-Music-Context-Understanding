"""Audio -> per-segment feature vectors, and the resumable HDF5 cache.

The canonical 96-dim node feature order is fixed by the data contract and is
asserted at build time. Do not reorder these blocks; three people are coding
against the indices.

    [  0: 16]  mel-band mean over 16 pooled mel bands
    [ 16: 32]  mel-band std  over 16 pooled mel bands
    [ 32: 52]  MFCC mean            (20)
    [ 52: 72]  MFCC std             (20)
    [ 72: 84]  chroma mean          (12)
    [ 84: 91]  spectral contrast mean (7 = n_bands + 1)
    [ 91: 96]  [centroid, bandwidth, rolloff, zcr, rms] mean (5)
                                                        ---- total 96
"""
from __future__ import annotations

import contextlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .utils import get_logger, resolve_path, save_json

LOGGER = get_logger("gbmc.audio")

__all__ = [
    "FEATURE_LAYOUT",
    "NODE_FEAT_DIM",
    "load_audio",
    "log_mel",
    "chroma_cqt",
    "mfcc",
    "segment_indices",
    "segment_features",
    "compute_norm_stats",
    "apply_norm",
    "extract_dataset",
    "feature_names",
    "verify_cache",
    "cache_keys_path",
    "pooled_log_mel",
    "compute_mel_stats",
]

N_POOLED_MEL_BANDS = 16

FEATURE_LAYOUT: list[tuple[str, int]] = [
    ("mel_mean", 16),
    ("mel_std", 16),
    ("mfcc_mean", 20),
    ("mfcc_std", 20),
    ("chroma_mean", 12),
    ("contrast_mean", 7),
    ("lowlevel_mean", 5),
]
NODE_FEAT_DIM = sum(size for _, size in FEATURE_LAYOUT)
assert NODE_FEAT_DIM == 96, "the data contract fixes the node feature dim at 96"


def feature_names() -> list[str]:
    """Human-readable name per feature index -- used by EDA and the report."""
    names: list[str] = []
    low_level = ["centroid", "bandwidth", "rolloff", "zcr", "rms"]
    for block, size in FEATURE_LAYOUT:
        if block == "lowlevel_mean":
            names += [f"{n}_mean" for n in low_level]
        else:
            names += [f"{block}_{i}" for i in range(size)]
    return names


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def load_audio(path, sr: int = 22050, mono: bool = True, duration=None,
               offset: float = 0.0) -> np.ndarray:
    """Decode audio to a float32 mono waveform at ``sr``.

    Raises on an undecodable file rather than returning silence -- a 0-byte
    yt-dlp artefact must fail loudly at manifest-build time, not become a track
    of zeros that quietly trains the model on nothing.

    ``path`` may be absolute or relative to the repository root. The committed
    manifests store the relative form so they work on any machine; a manifest
    built locally holds absolute paths and is passed through unchanged.
    """
    import librosa

    y, _ = librosa.load(str(resolve_path(path)), sr=sr, mono=mono,
                        duration=duration, offset=offset)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError(f"decoded zero samples from {path}")
    return y


def log_mel(y: np.ndarray, sr: int, cfg) -> np.ndarray:
    """``[n_mels, T]`` log-power mel spectrogram in dB."""
    import librosa

    audio_cfg = cfg["audio"] if "audio" in cfg else cfg
    spec = librosa.feature.melspectrogram(
        y=np.asarray(y, dtype=np.float32),
        sr=sr,
        n_fft=int(audio_cfg["n_fft"]),
        hop_length=int(audio_cfg["hop_length"]),
        n_mels=int(audio_cfg["n_mels"]),
        power=2.0,
    )
    return librosa.power_to_db(spec, ref=np.max).astype(np.float32)


def chroma_cqt(y: np.ndarray, sr: int, cfg) -> np.ndarray:
    """``[12, T]`` CQT chroma; falls back to STFT chroma on very short clips."""
    import librosa

    audio_cfg = cfg["audio"] if "audio" in cfg else cfg
    hop = int(audio_cfg["hop_length"])
    n_chroma = int(audio_cfg.get("n_chroma", 12))
    y = np.asarray(y, dtype=np.float32)
    try:
        return librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=hop, n_chroma=n_chroma
        ).astype(np.float32)
    except Exception:
        # CQT needs enough samples for its lowest filter; short excerpts fail.
        return librosa.feature.chroma_stft(
            y=y, sr=sr, hop_length=hop, n_fft=int(audio_cfg["n_fft"]), n_chroma=n_chroma
        ).astype(np.float32)


def mfcc(y: np.ndarray, sr: int, cfg) -> np.ndarray:
    """``[n_mfcc, T]`` MFCCs computed from the log-mel spectrogram."""
    import librosa

    audio_cfg = cfg["audio"] if "audio" in cfg else cfg
    mel_db = log_mel(y, sr, cfg)
    return librosa.feature.mfcc(
        S=mel_db, sr=sr, n_mfcc=int(audio_cfg["n_mfcc"])
    ).astype(np.float32)


def segment_indices(y: np.ndarray, sr: int, cfg) -> list[tuple[int, int]]:
    """Sample-index windows for each segment: fixed grid or beat-synchronous.

    Returns at least ``segmentation.min_nodes`` windows (by shrinking the window
    on short clips) and at most ``max_nodes`` (by uniform subsampling), so every
    graph in the batch has a workable node count.
    """
    seg_cfg = cfg["segmentation"] if "segmentation" in cfg else cfg
    audio_cfg = cfg["audio"] if "audio" in cfg else {}
    mode = str(seg_cfg.get("mode", "fixed"))
    window_s = float(seg_cfg.get("window_s", 3.0))
    overlap = float(seg_cfg.get("overlap", 0.5))
    min_nodes = int(seg_cfg.get("min_nodes", 4))
    max_nodes = int(seg_cfg.get("max_nodes", 32))
    n = int(np.asarray(y).shape[-1])

    if mode == "beat_sync":
        try:
            import librosa

            hop = int(audio_cfg.get("hop_length", 512))
            _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, units="frames")
            frames = librosa.frames_to_samples(beats, hop_length=hop)
            bounds = np.unique(np.concatenate([[0], frames, [n]])).astype(int)
            # group beats into bars of 4 so a node is a musical unit, not a beat
            bounds = bounds[:: max(1, 4)]
            if bounds[-1] != n:
                bounds = np.append(bounds, n)
            windows = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b - a > 0]
            if len(windows) >= min_nodes:
                return _clip_windows(windows, max_nodes)
            LOGGER.debug("beat tracking gave %d segments; falling back to fixed", len(windows))
        except Exception as exc:  # pragma: no cover - librosa edge cases
            LOGGER.debug("beat tracking failed (%s); falling back to fixed", exc)

    win = max(int(window_s * sr), 1)
    hop = max(int(win * (1.0 - min(max(overlap, 0.0), 0.95))), 1)
    if n < win * min_nodes / max(1.0, (1.0 - overlap) * (min_nodes - 1) + 1):
        # short clip: shrink the window so we still reach min_nodes segments
        win = max(int(n / max(1, (1.0 - overlap) * (min_nodes - 1) + 1)), 1)
        hop = max(int(win * (1.0 - min(max(overlap, 0.0), 0.95))), 1)

    starts = list(range(0, max(n - win + 1, 1), hop))
    windows = [(s, min(s + win, n)) for s in starts]
    windows = [(a, b) for a, b in windows if b > a]
    if not windows:
        windows = [(0, n)]
    while len(windows) < min_nodes:
        # pad by repeating the final window: better a duplicated node than a
        # graph too small for 2 rounds of message passing.
        windows.append(windows[-1])
    return _clip_windows(windows, max_nodes)


def _clip_windows(windows: list[tuple[int, int]], max_nodes: int) -> list[tuple[int, int]]:
    if len(windows) <= max_nodes:
        return windows
    keep = np.linspace(0, len(windows) - 1, max_nodes).round().astype(int)
    return [windows[i] for i in keep]


# --------------------------------------------------------------------------- #
# the 96-dim node feature
# --------------------------------------------------------------------------- #
def _pool_mel_bands(mel_db: np.ndarray, n_bands: int = N_POOLED_MEL_BANDS) -> np.ndarray:
    """Average the ``n_mels`` rows down to ``n_bands`` contiguous bands."""
    n_mels = mel_db.shape[0]
    edges = np.linspace(0, n_mels, n_bands + 1).round().astype(int)
    out = np.zeros((n_bands, mel_db.shape[1]), dtype=np.float32)
    for i in range(n_bands):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        out[i] = mel_db[lo:hi].mean(axis=0)
    return out


def segment_features(y: np.ndarray, sr: int, cfg) -> np.ndarray:
    """``[num_seg, 96]`` features in the canonical contract order."""
    import librosa

    audio_cfg = cfg["audio"] if "audio" in cfg else cfg
    hop = int(audio_cfg["hop_length"])
    n_fft = int(audio_cfg["n_fft"])
    n_bands = int(audio_cfg.get("n_bands", 6))
    y = np.asarray(y, dtype=np.float32)

    mel_db = log_mel(y, sr, cfg)
    pooled = _pool_mel_bands(mel_db)
    mfccs = mfcc(y, sr, cfg)
    chroma = chroma_cqt(y, sr, cfg)

    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)).astype(np.float32)
    contrast = librosa.feature.spectral_contrast(
        S=stft, sr=sr, n_bands=n_bands
    ).astype(np.float32)
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr).astype(np.float32)
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr).astype(np.float32)
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr).astype(np.float32)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
    rms = librosa.feature.rms(S=stft, frame_length=n_fft, hop_length=hop).astype(np.float32)

    frame_stacks = {
        "mel": pooled, "mfcc": mfccs, "chroma": chroma, "contrast": contrast,
        "centroid": centroid, "bandwidth": bandwidth, "rolloff": rolloff,
        "zcr": zcr.astype(np.float32), "rms": rms,
    }
    n_frames = min(v.shape[1] for v in frame_stacks.values())
    frame_stacks = {k: v[:, :n_frames] for k, v in frame_stacks.items()}

    windows = segment_indices(y, sr, cfg)
    feats = np.zeros((len(windows), NODE_FEAT_DIM), dtype=np.float32)
    for i, (start, end) in enumerate(windows):
        lo = min(int(start // hop), max(n_frames - 1, 0))
        hi = max(min(int(np.ceil(end / hop)), n_frames), lo + 1)
        sl = slice(lo, hi)
        blocks = [
            frame_stacks["mel"][:, sl].mean(axis=1),
            frame_stacks["mel"][:, sl].std(axis=1),
            frame_stacks["mfcc"][:, sl].mean(axis=1),
            frame_stacks["mfcc"][:, sl].std(axis=1),
            frame_stacks["chroma"][:, sl].mean(axis=1),
            frame_stacks["contrast"][:, sl].mean(axis=1),
            np.array([
                frame_stacks["centroid"][0, sl].mean(),
                frame_stacks["bandwidth"][0, sl].mean(),
                frame_stacks["rolloff"][0, sl].mean(),
                frame_stacks["zcr"][0, sl].mean(),
                frame_stacks["rms"][0, sl].mean(),
            ], dtype=np.float32),
        ]
        feats[i] = np.concatenate(blocks).astype(np.float32)

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    assert feats.shape[1] == NODE_FEAT_DIM, (
        f"feature block sizes disagree with the contract: got {feats.shape[1]}"
    )
    return feats


# --------------------------------------------------------------------------- #
# normalisation -- TRAIN SPLIT ONLY
# --------------------------------------------------------------------------- #
def compute_norm_stats(manifest, split: str = "train", cfg=None, h5_path=None,
                       max_tracks: int | None = None) -> dict:
    """Per-dimension mean/std over the **train split only**.

    Using val or test rows here leaks their distribution into training, which
    inflates every downstream number. The returned dict is persisted and reused
    verbatim for val and test -- see :func:`apply_norm`.
    """
    import pandas as pd

    if split != "train":
        raise ValueError(
            f"normalisation statistics must come from the train split, got {split!r}"
        )
    if isinstance(manifest, (str, Path)):
        manifest = pd.read_csv(resolve_path(manifest))

    if isinstance(manifest, pd.DataFrame):
        rows = manifest[manifest["split"] == "train"]
        keys = rows["track_id"].astype(str).tolist()
    else:  # already an iterable of feature arrays
        keys = None

    count = 0
    total = np.zeros(NODE_FEAT_DIM, dtype=np.float64)
    total_sq = np.zeros(NODE_FEAT_DIM, dtype=np.float64)

    def _accumulate(arr: np.ndarray) -> None:
        nonlocal count, total, total_sq
        arr = np.asarray(arr, dtype=np.float64).reshape(-1, NODE_FEAT_DIM)
        count += arr.shape[0]
        total += arr.sum(axis=0)
        total_sq += (arr**2).sum(axis=0)

    if keys is not None:
        if h5_path is None:
            raise ValueError("compute_norm_stats needs h5_path when given a manifest")
        import h5py

        with h5py.File(resolve_path(h5_path), "r") as store:
            for i, key in enumerate(keys):
                if max_tracks and i >= max_tracks:
                    break
                if key in store:
                    # read one track at a time; never materialise the cache
                    _accumulate(store[key][...])
    else:
        for i, arr in enumerate(manifest):
            if max_tracks and i >= max_tracks:
                break
            _accumulate(arr)

    if count == 0:
        raise RuntimeError("no train-split features found to compute statistics from")

    mean = total / count
    var = np.maximum(total_sq / count - mean**2, 0.0)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0          # constant dimensions must not explode
    return {
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "n_segments": int(count),
        "split": "train",
        "dim": NODE_FEAT_DIM,
    }


def compute_mel_stats(manifest, mel_h5, split: str = "train",
                      max_tracks: int | None = 4000) -> dict:
    """Scalar mean/std of the log-mel cache over the **train split only**.

    A single scalar rather than per-band: the CNN sees the mel patch as an image
    and BatchNorm handles per-channel scale after the first layer. The point here
    is only to bring the input into unit range before the first convolution.
    """
    import h5py
    import pandas as pd

    if split != "train":
        raise ValueError(
            f"mel normalisation statistics must come from the train split, got {split!r}"
        )
    if isinstance(manifest, (str, Path)):
        manifest = pd.read_csv(resolve_path(manifest))
    train = manifest[manifest["split"] == "train"]

    # mel_h5 may be a single path or a {dataset: path} mapping, because the
    # caches are written per corpus; looking in the wrong one silently finds
    # nothing, which is how this failed the first time.
    caches = mel_h5 if isinstance(mel_h5, dict) else {"__single__": mel_h5}
    total = total_sq = 0.0
    count = 0
    for name, path in caches.items():
        path = resolve_path(path)
        if not path.exists():
            continue
        rows = (train if name == "__single__"
                else train[train.get("dataset", "") == name])
        keys = rows["track_id"].astype(str).tolist()
        if not keys:
            continue
        with h5py.File(path, "r") as store:
            for i, key in enumerate(keys):
                if max_tracks and i >= max_tracks:
                    break
                if key not in store:
                    continue
                arr = np.asarray(store[key][...], dtype=np.float64)
                total += float(arr.sum())
                total_sq += float((arr**2).sum())
                count += arr.size
    if count == 0:
        raise RuntimeError(
            f"no train-split mel patches found in {list(caches)} -- check that the "
            "mel cache matches the corpora in the manifest"
        )

    mean = total / count
    std = float(np.sqrt(max(total_sq / count - mean**2, 1e-12)))
    return {"mean": float(mean), "std": std, "n_values": int(count),
            "split": "train", "n_tracks_used": min(len(keys), max_tracks or len(keys))}


def apply_norm(feats, stats: dict) -> np.ndarray:
    """Standardise features with statistics computed on train."""
    arr = np.asarray(feats, dtype=np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if arr.shape[-1] != mean.shape[0]:
        raise ValueError(
            f"norm stats are {mean.shape[0]}-dim but features are {arr.shape[-1]}-dim"
        )
    return ((arr - mean) / np.maximum(std, 1e-6)).astype(np.float32)


# --------------------------------------------------------------------------- #
# the HDF5 cache
# --------------------------------------------------------------------------- #
def pooled_log_mel(y: np.ndarray, sr: int, cfg, n_frames: int = 256) -> np.ndarray:
    """Whole-track log-mel, time-pooled to exactly ``n_frames`` columns.

    Sized for fairness rather than fidelity. The CNN baseline must see the same
    span of music as the GNN, so this covers the **whole** track; storing it at
    native resolution would cost ~8 GB for MTAT alone, to serve a model that
    pools over time anyway. At 256 frames over 30 s the CNN gets ~0.12 s
    resolution against 32 GNN segments at ~1 s, so the baseline is if anything
    favoured -- the right direction for a fair comparison.
    """
    mel = log_mel(y, sr, cfg)                       # [n_mels, T]
    width = mel.shape[1]
    if width == 0:
        return np.zeros((mel.shape[0], int(n_frames)), dtype=np.float32)
    edges = np.linspace(0, width, int(n_frames) + 1).round().astype(int)
    out = np.empty((mel.shape[0], int(n_frames)), dtype=np.float32)
    for i in range(int(n_frames)):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        out[:, i] = mel[:, lo:min(hi, width)].mean(axis=1)
    return out


def _extract_one(args) -> tuple[str, "np.ndarray | None", "np.ndarray | None", str]:
    """Worker: decode one file once and return both cached views of it.

    Decoding dominates the cost (25k mp3 decodes for MTAT), so the segment
    features and the CNN baseline mel patch are produced from a single decode
    rather than two passes over the corpus.
    """
    track_id, audio_path, cfg, sr, max_duration, want_mel, mel_frames = args
    try:
        y = load_audio(audio_path, sr=sr, mono=True, duration=max_duration)
        feats = segment_features(y, sr, cfg)
        mel = (pooled_log_mel(y, sr, cfg, mel_frames).astype(np.float16)
               if want_mel else None)
        return str(track_id), feats.astype(np.float16), mel, "ok"
    except Exception as exc:
        return str(track_id), None, None, f"{type(exc).__name__}: {exc}"


def extract_dataset(manifest, out_h5, cfg, n_workers: int = 6,
                    overwrite: bool = False, log_path=None, mel_h5=None,
                    mel_frames: int = 256) -> dict:
    """Extract features for a manifest into a resumable float16 HDF5 cache.

    Resumability is not a nicety: 25k MTAT clips take hours on 6 cores and this
    *will* be interrupted. Keys already present are skipped, and each result is
    flushed as it arrives so a kill -9 costs at most one track.

    float16 halves the cache (~1.5 GB -> ~750 MB for MTAT) and the values are
    segment-level statistics of dB-scale features, where fp16's ~3 decimal
    digits are far below the noise floor of the estimates themselves.
    """
    import h5py
    import pandas as pd

    if isinstance(manifest, (str, Path)):
        manifest = pd.read_csv(resolve_path(manifest))
    out_path = resolve_path(out_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audio_cfg = cfg["audio"]
    sr = int(audio_cfg["sample_rate"])
    max_duration = float(audio_cfg.get("max_duration_s", 30.0)) or None
    cfg_plain = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)

    integrity = verify_cache(out_path)
    if integrity["exists"] and not integrity["readable"] and not overwrite:
        raise RuntimeError(
            f"{out_path} exists but cannot be opened ({integrity['error']}). "
            "Delete it and re-run, or pass overwrite=True; see the sidecar for "
            "what had completed."
        )

    mode = "w" if overwrite else "a"
    stats = {"total": int(len(manifest)), "written": 0, "skipped": 0, "failed": 0}
    failures: list[dict] = []
    mel_path = resolve_path(mel_h5) if mel_h5 else None
    if mel_path is not None:
        mel_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, mode) as store, \
            (h5py.File(mel_path, mode) if mel_path is not None
             else contextlib.nullcontext()) as mel_store:
        existing = set(store.keys())
        todo = []
        for row in manifest.itertuples(index=False):
            key = str(row.track_id)
            if key in existing:
                stats["skipped"] += 1
                continue
            todo.append((key, row.audio_path, cfg_plain, sr, max_duration,
                         mel_store is not None, int(mel_frames)))

        LOGGER.info(
            "extract_dataset: %d tracks, %d already cached, %d to do -> %s",
            stats["total"], stats["skipped"], len(todo), out_path,
        )
        if not todo:
            store.attrs["feature_dim"] = NODE_FEAT_DIM
            _write_key_sidecar(out_path, set(store.keys()))
            return stats

        n_workers = max(1, int(n_workers))
        if n_workers == 1 or len(todo) < 4:
            results = (_extract_one(item) for item in todo)
            for key, feats, mel, status in results:
                _write_result(store, key, feats, status, stats, failures, mel_store, mel)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_extract_one, item): item[0] for item in todo}
                for future in as_completed(futures):
                    key, feats, mel, status = future.result()
                    _write_result(store, key, feats, status, stats, failures,
                                  mel_store, mel)
                    if stats["written"] % 200 == 0:
                        _write_key_sidecar(out_path, set(store.keys()))
                        LOGGER.info("  %d written, %d failed, %d to go",
                                    stats["written"], stats["failed"],
                                    len(todo) - stats["written"] - stats["failed"])

        store.attrs["feature_dim"] = NODE_FEAT_DIM
        store.attrs["sample_rate"] = sr
        store.attrs["layout"] = json.dumps(FEATURE_LAYOUT)
        if mel_store is not None:
            mel_store.attrs["n_mels"] = int(cfg["audio"]["n_mels"])
            mel_store.attrs["n_frames"] = int(mel_frames)
            mel_store.attrs["pooling"] = "whole track, mean-pooled to n_frames"
        final_keys = set(store.keys())
    _write_key_sidecar(out_path, final_keys)

    if log_path:
        save_json({"stats": stats, "failures": failures}, log_path)
    LOGGER.info("extract_dataset done: %s", stats)
    return stats


def _write_result(store, key, feats, status, stats, failures,
                  mel_store=None, mel=None) -> None:
    if feats is None:
        stats["failed"] += 1
        failures.append({"track_id": key, "error": status})
        return
    if key in store:  # pragma: no cover - concurrent re-run
        del store[key]
    store.create_dataset(key, data=feats, dtype="float16", compression="lzf")
    if mel_store is not None and mel is not None:
        if key in mel_store:  # pragma: no cover - concurrent re-run
            del mel_store[key]
        mel_store.create_dataset(key, data=mel, dtype="float16", compression="lzf")
        mel_store.flush()
    # Flush per item, not per batch. HDF5 append is not transactional, so the
    # only bound on damage from a kill -9 is how much sits unflushed; one track
    # is an acceptable loss, two hundred is not.
    store.flush()
    stats["written"] += 1


def cache_keys_path(h5_path) -> Path:
    """Sidecar listing the keys known-good in an HDF5 cache."""
    out = resolve_path(h5_path)
    return out.with_name(out.name + ".keys.json")


def _write_key_sidecar(h5_path, keys) -> None:
    """Record completed keys atomically, outside the HDF5 file.

    HDF5 has no transactional guarantee, so a kill during a write can leave the
    container unopenable and take every extracted track with it. The sidecar is
    written with the atomic temp-then-rename dance, so even in that case we know
    exactly what had completed and can say so instead of guessing.
    """
    save_json({"h5": str(resolve_path(h5_path)), "n_keys": len(keys),
               "keys": sorted(keys)}, cache_keys_path(h5_path))


def verify_cache(h5_path) -> dict:
    """Open a cache read-only and report what is actually in it.

    Called at the top of :func:`extract_dataset` so a corrupt container is
    reported with a recovery instruction rather than surfacing later as a
    confusing KeyError in the dataloader.
    """
    import h5py

    path = resolve_path(h5_path)
    report = {"path": str(path), "exists": path.exists(), "readable": False,
              "n_keys": 0, "feature_dim": None, "sidecar_keys": None, "error": None}
    if not path.exists():
        return report

    sidecar = cache_keys_path(path)
    if sidecar.exists():
        try:
            report["sidecar_keys"] = int(json.loads(
                sidecar.read_text(encoding="utf-8"))["n_keys"])
        except Exception:
            report["sidecar_keys"] = None

    try:
        with h5py.File(path, "r") as store:
            report["readable"] = True
            report["n_keys"] = len(store.keys())
            report["feature_dim"] = int(store.attrs.get("feature_dim", -1))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        LOGGER.error(
            "feature cache %s is unreadable (%s). Extraction is item-level "
            "resumable, so the fix is to delete it and re-run -- but check the "
            "sidecar %s first: it records %s completed keys, which is what you "
            "are about to redo.",
            path, report["error"], sidecar.name,
            report["sidecar_keys"] if report["sidecar_keys"] is not None else "an unknown number of",
        )
    return report


def main(argv=None) -> int:
    """CLI: extract the feature cache for one or more datasets.

    One HDF5 per dataset, not one shared file. Three reasons: a corrupt container
    then costs one corpus rather than all of them; per-dataset normalisation
    statistics are what the contract actually calls for; and the datasets finish
    at wildly different times (MTAT is hours, DEAM is minutes), so separate files
    let downstream work start on whichever is ready.
    """
    import argparse

    import pandas as pd

    from .splits import assert_no_leakage
    from .utils import load_config, parse_overrides

    parser = argparse.ArgumentParser(
        description="Extract the 96-dim segment feature cache (resumable)."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--datasets", nargs="*",
                        default=["mtat", "fma", "deam", "musiccaps"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="extract only the first N rows (for a quick trial run)")
    parser.add_argument("--no-mel", action="store_true",
                        help="skip the CNN-baseline mel cache")
    parser.add_argument("--mel-frames", type=int, default=256)
    parser.add_argument("--skip-norm-stats", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args(argv)

    cfg = load_config(args.config, parse_overrides(args.override))
    splits_dir = resolve_path(cfg["paths"]["splits"])
    processed = resolve_path(cfg["paths"]["processed"])
    processed.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    for name in args.datasets:
        manifest_path = splits_dir / f"{name}_manifest.csv"
        if not manifest_path.exists():
            LOGGER.warning("no manifest for %s at %s -- run `make splits` first",
                           name, manifest_path)
            continue
        manifest = pd.read_csv(manifest_path)
        if "dataset" not in manifest.columns:
            manifest["dataset"] = name
        # cheap insurance: never extract from a manifest that leaks
        assert_no_leakage(manifest)
        if args.limit:
            manifest = manifest.head(int(args.limit))

        out_h5 = processed / f"features_{name}.h5"
        mel_h5 = None if args.no_mel else processed / f"mels_{name}.h5"
        LOGGER.info("extracting %s: %d tracks -> %s", name, len(manifest), out_h5)

        stats = extract_dataset(
            manifest, out_h5, cfg, n_workers=args.workers, overwrite=args.overwrite,
            log_path=processed / f"extract_log_{name}.json",
            mel_h5=mel_h5, mel_frames=args.mel_frames,
        )

        # normalisation statistics come from the TRAIN split only, and the
        # provenance is recorded so a later run can assert it rather than trust it.
        if not args.skip_norm_stats:
            try:
                norm = compute_norm_stats(manifest, split="train", cfg=cfg,
                                          h5_path=out_h5)
                norm["dataset"] = name
                norm["source_manifest"] = str(manifest_path)
                norm["n_train_tracks"] = int((manifest["split"] == "train").sum())
                save_json(norm, processed / f"norm_stats_{name}.json")
                stats["norm_segments"] = norm["n_segments"]
                LOGGER.info("%s norm stats: %d train segments, %d dims (train split only)",
                            name, norm["n_segments"], norm["dim"])
            except Exception as exc:
                LOGGER.warning("could not compute %s normalisation statistics: %s",
                               name, exc)

        summary[name] = stats
        LOGGER.info("%s done: %s", name,
                    {k: v for k, v in stats.items() if k != "failures"})

    save_json(summary, processed / "extract_summary.json")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
