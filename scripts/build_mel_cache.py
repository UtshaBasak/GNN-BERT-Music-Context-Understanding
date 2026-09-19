#!/usr/bin/env python
"""build a **full-resolution** log-mel cache for the CNN baseline.

    python scripts/build_mel_cache.py --datasets mtat,fma --workers 6

The original cache (`mels_{corpus}.h5`) mean-pools every track to 256 frames,
which is ~8.8 frames/second. A 3-second excerpt is then 26 columns wide -- far
too coarse for a convolutional stack whose whole premise is local time-frequency
structure. That pooling is the most likely reason B2 scored 0.165 macro-F1 on
MTAT when published mel-CNNs reach 0.38-0.41.

This writes the mel at native resolution instead: 22050 Hz / hop 512 = 43.07
frames per second, so a 3 s chunk is 129 columns. Storage is the reason the
pooled cache existed, so it is handled rather than avoided -- float16 plus the
HDF5 shuffle filter and gzip-4 measures 1.63x on this data, taking MTAT from
6.9 GB to ~4.2 GB and FMA from 2.7 GB to ~1.6 GB, at 2.2 ms per clip to read
back.

Resumable: keys already present are skipped, so an interrupted run costs at
most one track.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio_features import load_audio, log_mel  # noqa: E402
from src.utils import get_logger, load_config, resolve_path, save_json  # noqa: E402

LOGGER = get_logger("gbmc.melcache")

COMPRESSION = dict(compression="gzip", compression_opts=4, shuffle=True)


def _one(args):
    track_id, audio_path, cfg, sr, max_duration = args
    try:
        y = load_audio(audio_path, sr=sr, mono=True, duration=max_duration)
        mel = log_mel(y, sr, cfg).astype(np.float16)
        if mel.ndim != 2 or mel.shape[1] < 8:
            return str(track_id), None, f"degenerate mel {mel.shape}"
        return str(track_id), mel, "ok"
    except Exception as exc:                                   # noqa: BLE001
        return str(track_id), None, f"{type(exc).__name__}: {exc}"


def build(name: str, cfg, workers: int, overwrite: bool) -> dict:
    import h5py
    import pandas as pd

    manifest = pd.read_csv(resolve_path(cfg["paths"]["splits"]) / f"{name}_manifest.csv")
    out = resolve_path(cfg["paths"]["processed"]) / f"mels_full_{name}.h5"
    out.parent.mkdir(parents=True, exist_ok=True)

    audio = cfg["audio"]
    sr = int(audio["sample_rate"])
    max_duration = float(audio.get("max_duration_s", 30.0)) or None
    plain = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)

    stats = {"dataset": name, "total": len(manifest), "written": 0,
             "skipped": 0, "failed": 0}
    failures = []
    started = time.time()

    with h5py.File(out, "w" if overwrite else "a") as store:
        existing = set(store.keys())
        todo = []
        for row in manifest.itertuples(index=False):
            key = str(row.track_id)
            if key in existing:
                stats["skipped"] += 1
                continue
            todo.append((key, row.audio_path, plain, sr, max_duration))

        LOGGER.info("%s: %d tracks, %d cached, %d to do -> %s",
                    name, stats["total"], stats["skipped"], len(todo), out.name)

        def write(key, mel, status):
            if mel is None:
                stats["failed"] += 1
                failures.append({"track_id": key, "error": status})
                return
            if key in store:
                del store[key]
            store.create_dataset(key, data=mel, **COMPRESSION)
            stats["written"] += 1
            if stats["written"] % 500 == 0:
                store.flush()
                rate = stats["written"] / max(time.time() - started, 1e-6)
                left = (len(todo) - stats["written"]) / max(rate, 1e-6)
                LOGGER.info("  %s %d/%d written (%.1f/s, ~%.0f min left)",
                            name, stats["written"], len(todo), rate, left / 60)

        if workers <= 1 or len(todo) < 4:
            for item in todo:
                write(*_one(item))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_one, item) for item in todo]
                for future in as_completed(futures):
                    write(*future.result())

        store.attrs["n_mels"] = int(audio["n_mels"])
        store.attrs["sample_rate"] = sr
        store.attrs["hop_length"] = int(audio["hop_length"])
        store.attrs["n_fft"] = int(audio["n_fft"])
        store.attrs["frames_per_second"] = sr / float(audio["hop_length"])
        store.attrs["pooling"] = "none -- native resolution"
        stats["n_keys"] = len(store.keys())

    stats["seconds"] = round(time.time() - started, 1)
    stats["size_gb"] = round(out.stat().st_size / 1e9, 3)
    save_json({"stats": stats, "failures": failures[:200]},
              resolve_path(cfg["paths"]["processed"]) / f"mel_full_log_{name}.json")
    LOGGER.info("%s done: %s", name, stats)
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="mtat,fma")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    summary = {}
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        summary[name] = build(name, cfg, args.workers, args.overwrite)
    save_json(summary, resolve_path(cfg["paths"]["processed"]) / "mel_full_summary.json")
    for name, stats in summary.items():
        print(f"{name}: {stats['n_keys']} keys, {stats['size_gb']} GB, "
              f"{stats['failed']} failed, {stats['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
