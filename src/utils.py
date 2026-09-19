"""Shared plumbing: seeding, config, device selection, AMP helpers, logging.

Everything that more than one module needs and that has no scientific content
lives here so the rest of the codebase stays about music.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

# torch is imported lazily, inside the functions that need it. On Windows a
# ProcessPoolExecutor worker re-imports the module that defines its task
# function, and src.audio_features imports this module -- so a module-level
# torch import meant six extraction workers each loading torch's DLLs, which
# exhausts the commit limit (WinError 1455: the paging file is too small).
# The extraction workers need numpy and librosa; they never touch torch.

__all__ = [
    "set_seed",
    "seed_worker",
    "get_generator",
    "load_config",
    "get_device",
    "count_parameters",
    "autocast_ctx",
    "log_vram",
    "find_checkpoint",
    "vocabulary_hash",
    "encoder_name_from_checkpoint",
    "get_logger",
    "AttrDict",
    "save_json",
    "load_json",
    "ensure_dir",
    "resolve_path",
    "project_root",
    "atomic_write_text",
    "atomic_write_bytes",
    "atomic_torch_save",
    "SYNTHETIC",
    "REAL",
    "detect_provenance",
    "guard_against_synthetic",
    "SyntheticArtifactError",
]

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "gbmc", level: int = logging.INFO) -> logging.Logger:
    """Return a module logger with a single stdout handler (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


LOGGER = get_logger()


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed every RNG this project can reach, including subprocess hashing."""
    import torch

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cudnn autotuning picks different algorithms run to run; pin it down.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """`worker_init_fn` for DataLoader: derive each worker seed from torch's."""
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int):
    """Seeded generator to hand to DataLoader(generator=...)."""
    import torch

    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
class AttrDict(dict):
    """dict that also supports attribute access, recursively.

    Lets config be written either way (cfg["gnn"]["conv"] or cfg.gnn.conv),
    which matters because the spec signatures use cfg.graph.node_feat_dim.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            super().__setitem__(key, self._wrap(value))

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, AttrDict):
            return value
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - error path
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __setitem__(self, key, value):
        super().__setitem__(key, self._wrap(value))

    def to_dict(self) -> dict:
        out: dict = {}
        for k, v in self.items():
            if isinstance(v, AttrDict):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, AttrDict) else x for x in v]
            else:
                out[k] = v
        return out


def _coerce(text: str) -> Any:
    """Turn a CLI string into the most specific type it parses as."""
    lowered = text.strip().lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return yaml.safe_load(text)
    except Exception:  # pragma: no cover - yaml is very permissive
        return text


def _set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], Mapping):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def load_config(path: str = "config.yaml", overrides: dict | None = None) -> "AttrDict":
    """Load YAML config and apply dotted-key overrides.

    ``overrides`` accepts either {"train.epochs": 1} or {"train": {"epochs": 1}};
    values may be raw strings (from argparse) and are coerced by YAML rules.
    """
    cfg_path = Path(path)
    if not cfg_path.exists() and not cfg_path.is_absolute():
        cfg_path = project_root() / path
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if overrides:
        for key, value in overrides.items():
            if isinstance(value, str):
                value = _coerce(value)
            if "." in key:
                _set_nested(raw, key, value)
            elif isinstance(value, Mapping) and isinstance(raw.get(key), Mapping):
                raw[key].update(value)
            else:
                raw[key] = value

    cfg = AttrDict(raw)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def parse_overrides(pairs: Iterable[str] | None) -> dict:
    """["train.epochs=1", "gnn.conv=gatv2"] -> {"train.epochs": 1, ...}."""
    out: dict = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--override expects key=value, got {item!r}")
        key, _, value = item.partition("=")
        out[key.strip()] = _coerce(value)
    return out


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def project_root() -> Path:
    """Repository root (the directory holding src/)."""
    return Path(__file__).resolve().parent.parent


def resolve_path(p: "str | os.PathLike") -> Path:
    """Resolve a config-relative path against the repo root."""
    path = Path(p)
    return path if path.is_absolute() else (project_root() / path)


def ensure_dir(p: "str | os.PathLike") -> Path:
    path = resolve_path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitise(obj: Any) -> Any:
    """Replace non-finite floats with ``None`` so the output is valid JSON.

    ``json.dump`` happily writes bare ``NaN``/``Infinity``, which Python reads
    back but every strict parser rejects. A
    missing metric is genuinely null, so this is also the more honest encoding.
    """
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    return obj


def atomic_write_text(path: "str | os.PathLike", text: str,
                      encoding: str = "utf-8") -> Path:
    """Write via ``path.tmp`` -> fsync -> ``os.replace``.

    A half-written manifest or state file that *looks* complete is the worst
    failure mode in a multi-session pipeline: the next session reads it, believes
    it, and builds on corrupt input. ``os.replace`` is atomic on both POSIX and
    Windows (same volume), so a reader sees either the old file or the new one,
    never a partial one. The ``fsync`` before the rename is what makes that hold
    across a power loss rather than merely across a process kill.
    """
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding=encoding, newline="") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)
    return out


def atomic_write_bytes(path: "str | os.PathLike", payload: bytes) -> Path:
    """Binary counterpart of :func:`atomic_write_text`."""
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)
    return out


def atomic_torch_save(obj: Any, path: "str | os.PathLike") -> Path:
    """``torch.save`` through a temp file, so a killed run leaves no half-checkpoint."""
    import io as _io

    import torch

    buffer = _io.BytesIO()
    torch.save(obj, buffer)
    return atomic_write_bytes(path, buffer.getvalue())


def save_json(obj: Any, path: "str | os.PathLike", indent: int = 2) -> Path:
    """Serialise to JSON atomically, with non-finite floats encoded as ``null``."""
    text = json.dumps(_sanitise(obj), indent=indent, default=_json_default,
                      allow_nan=False)
    return atomic_write_text(path, text)


def load_json(path: "str | os.PathLike") -> Any:
    with open(resolve_path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return _sanitise(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, AttrDict):
        return _sanitise(obj.to_dict())
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):      # a torch.Tensor
        return _sanitise(obj.detach().cpu().tolist())
    return str(obj)


# --------------------------------------------------------------------------- #
# device / AMP / VRAM
# --------------------------------------------------------------------------- #
def get_device(pref: str = "cuda"):
    """Honour the preference, failing fast on a GPU this torch cannot drive.

    ``torch.cuda.is_available()`` returns True for a GPU whose architecture the
    installed torch was never compiled for; the failure only surfaces later as
    ``CUDA error: no kernel image is available for execution on the device``,
    once per run, with no hint about the cause. Kaggle hits this exactly: its
    P100 accelerator is sm_60 (Pascal) and recent torch builds ship sm_70 and
    up, so every run in a sweep dies the same cryptic way.

    Checking the capability up front turns five confusing failures into one
    actionable message.
    """
    import torch

    pref = (pref or "cuda").lower()
    if not pref.startswith("cuda"):
        return torch.device(pref)
    if not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")

    index = 0 if ":" not in pref else int(pref.split(":")[1])
    major, minor = torch.cuda.get_device_capability(index)
    capability = major * 10 + minor
    built_for = sorted(
        int(arch.split("_")[1])
        for arch in torch.cuda.get_arch_list()
        if arch.startswith("sm_")
    )
    # Only the "too old" case is a hard failure: a device newer than anything
    # listed can usually JIT from the embedded PTX.
    if built_for and capability < min(built_for):
        name = torch.cuda.get_device_name(index)
        archs = ", ".join(f"sm_{a}" for a in built_for)
        raise RuntimeError(
            f"{name} is compute capability sm_{capability}, but this PyTorch "
            f"({torch.__version__}) only has kernels for sm_{min(built_for)} and "
            f"newer ({archs}).\n"
            "Every CUDA call would fail with 'no kernel image is available for "
            "execution on the device'.\n"
            "On Kaggle: Settings -> Accelerator -> GPU T4 x2 (sm_75). The P100 "
            "is sm_60 and is no longer supported by current torch builds.\n"
            "Otherwise pass --device cpu, or install a torch built for this GPU."
        )
    return torch.device(pref)


def count_parameters(model, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def autocast_ctx(enabled: bool, device_type: str = "cuda", dtype=None):
    """torch.amp.autocast wrapper that degrades gracefully off-GPU.

    The GTX 1650 (TU117, sm_75) has no tensor cores, so AMP here buys memory,
    not speed -- which is exactly what a 4 GB card needs. bf16 is unsupported
    on sm_75, so fp16 is the only useful autocast dtype.
    """
    import torch

    if not enabled or device_type != "cuda" or not torch.cuda.is_available():
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=dtype or torch.float16)


def make_grad_scaler(enabled: bool, device_type: str = "cuda"):
    """GradScaler that is a no-op when AMP is off or we are on CPU."""
    import torch

    use = bool(enabled) and device_type == "cuda" and torch.cuda.is_available()
    return torch.amp.GradScaler("cuda", enabled=use)


def log_vram(tag: str = "", logger: "logging.Logger | None" = None) -> dict:
    """Report current/peak CUDA allocation -- the 4 GB debugging workhorse."""
    import torch

    logger = logger or LOGGER
    if not torch.cuda.is_available():
        return {"tag": tag, "allocated_mb": 0.0, "peak_mb": 0.0, "reserved_mb": 0.0}
    stats = {
        "tag": tag,
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
    }
    logger.info(
        "VRAM[%s] allocated=%.1f MB peak=%.1f MB reserved=%.1f MB",
        tag, stats["allocated_mb"], stats["peak_mb"], stats["reserved_mb"],
    )
    return stats


def reset_vram_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# provenance: keeping synthetic artifacts out of the report
# --------------------------------------------------------------------------- #
SYNTHETIC = "synthetic"
REAL = "real"

#: values that mark an artifact as not-real, whichever field they appear in
_SYNTHETIC_MARKERS = {"synthetic", "syn", "fake", "smoke"}


class SyntheticArtifactError(RuntimeError):
    """Raised when synthetic data reaches a code path meant for real results.

    This is a hard failure on purpose. A synthetic checkpoint silently loaded by
    ``evaluate.py`` produces tables and figures that are indistinguishable from
    real ones at a glance, and that is a submission hazard, not an inconvenience.
    """


def detect_provenance(obj) -> str:
    """Return ``"synthetic"`` or ``"real"`` for a graph, manifest, dict or path.

    Provenance is an explicit field rather than an inference: the synthetic
    generator deliberately reuses the real corpus names in ``dataset`` (so that
    the ``-1`` / ``nan`` sentinel pattern matches reality), which means the
    corpus name cannot be used to tell fake from real. ``dataset == "synthetic"``
    is still honoured for any artifact written before the field existed.
    """
    if obj is None:
        return REAL

    if isinstance(obj, (str, os.PathLike)):
        path = Path(obj)
        if any(part.lower() in _SYNTHETIC_MARKERS or
               part.lower().startswith("_synthetic") for part in path.parts):
            return SYNTHETIC
        if path.is_file() and path.suffix == ".json":
            try:
                return detect_provenance(load_json(path))
            except Exception:  # pragma: no cover - unreadable sidecar
                return REAL
        return REAL

    for field in ("provenance", "dataset", "source"):
        value = None
        if isinstance(obj, dict):
            value = obj.get(field)
        elif hasattr(obj, field):
            value = getattr(obj, field)
        if isinstance(value, str) and value.strip().lower() in _SYNTHETIC_MARKERS:
            return SYNTHETIC

    if isinstance(obj, dict) and obj.get("synthetic") is True:
        return SYNTHETIC
    if hasattr(obj, "columns"):                      # a pandas DataFrame
        for column in ("provenance", "dataset"):
            if column in obj.columns:
                values = {str(v).strip().lower() for v in obj[column].unique()}
                if values & _SYNTHETIC_MARKERS:
                    return SYNTHETIC
    if isinstance(obj, (list, tuple)):
        return SYNTHETIC if any(detect_provenance(o) == SYNTHETIC for o in obj) else REAL
    return REAL


def guard_against_synthetic(obj, allow_synthetic: bool, context: str = "") -> str:
    """Refuse to proceed on synthetic input unless it was asked for explicitly.

    ``allow_synthetic`` is the caller's ``--synthetic`` flag. Passing it makes
    the synthetic path legal; omitting it makes synthetic input an error rather
    than a silent source of fake results.
    """
    provenance = detect_provenance(obj)
    if provenance == SYNTHETIC and not allow_synthetic:
        raise SyntheticArtifactError(
            f"refusing to run on synthetic data{f' ({context})' if context else ''}. "
            "These artifacts came from `python -m src.synthetic` and must never "
            "reach a reported table or figure. Pass --synthetic if that is "
            "genuinely what you want, or point this at real data."
        )
    if provenance == REAL and allow_synthetic:
        LOGGER.warning(
            "--synthetic was passed but %s looks like real data; continuing.",
            context or "the input",
        )
    return provenance


def find_checkpoint(ckpt_dir, task: int, seed: int, run_tag: str | None = None,
                    kind: str = "best"):
    """Locate a checkpoint, preferring an exact run tag.

    Checkpoints are written as ``task{N}_seed{S}[_{tag}]_{kind}.pt``. Readers
    ask for a tag when they need a specific model -- the case studies need the
    MusicCaps fusion run, not whichever fusion mode trained last -- and get a
    clear ``None`` rather than the wrong weights when it is absent.

    Search order: the exact tag, then the untagged name (artifacts written
    before tagging existed), then any tagged file for that task and seed, newest
    first, so a directory that only holds tagged runs still resolves.
    """
    from pathlib import Path as _P

    directory = _P(ckpt_dir)
    if not directory.exists():
        return None
    if run_tag:
        exact = directory / f"task{task}_seed{seed}_{run_tag}_{kind}.pt"
        if exact.exists():
            return exact
    plain = directory / f"task{task}_seed{seed}_{kind}.pt"
    if plain.exists():
        return plain
    candidates = sorted(directory.glob(f"task{task}_seed{seed}_*_{kind}.pt"),
                        key=lambda q: q.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def vocabulary_hash(tags) -> str:
    """Stable 12-hex digest of an ordered tag vocabulary.

    Ordered on purpose. The vocabulary order fixes which column belongs to which
    tag, so two runs sharing a set but not an order produce per-tag tables that
    disagree while their aggregate metrics look identical -- the kind of
    mismatch that survives review because nothing about it looks wrong.

    Used to decide whether a result computed elsewhere is comparable with the
    local ones. A short digest rather than the full list because it travels in
    every result payload.
    """
    import hashlib

    if not tags:
        return ""
    joined = "\u0000".join(str(t) for t in tags)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def encoder_name_from_checkpoint(payload, fallback: str) -> str:
    """The text encoder a checkpoint was trained with, not the current default.

    ``config.yaml`` defaults ``bert.model_name`` to distilbert, while the Task 3
    and Task 4 headline runs use bert-base. Building the encoder from the config
    and then calling ``load_state_dict(strict=False)`` -- or the shape-tolerant
    loader in evaluate.py -- silently leaves most of a 12-layer tower randomly
    initialised and returns numbers that look entirely reasonable. It showed up
    three times in this project as "96 parameter(s) missing" and "restored
    48/144 tensors", neither of which stops anything.

    Every checkpoint stores the config it was trained under, so the architecture
    is recoverable rather than guessable. Callers should pass the result into
    the encoder they build.
    """
    if not isinstance(payload, Mapping):
        return fallback
    config = payload.get("config") or payload.get("cfg") or {}
    if not isinstance(config, Mapping):
        return fallback
    bert = config.get("bert") or {}
    name = bert.get("model_name") if isinstance(bert, Mapping) else None
    return str(name) if name else fallback
