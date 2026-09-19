"""Unified training entry point for all four tasks.

    python -m src.train --task {1,2,3,4} --config config.yaml [--synthetic]
                        [--seed N] [--override key=value ...]
                        [--device cuda|cpu] [--dry-run]
                        [--precompute-text-embeddings]

Every run: seeds globally, asserts split disjointness before touching a model,
trains under AMP with gradient accumulation, checkpoints each epoch, early-stops
on the configured metric, and writes ``results/task{N}_seed{S}.json``.

Threshold discipline, enforced structurally rather than by convention: per-tag
thresholds are fitted on the *validation* scores at the end of training, frozen
into the result payload, and only then applied to test. The test split is
touched exactly once, after the model and the thresholds are both final.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import metrics as M
from .bert_encoder import (FREEZE_MODES, BertTagClassifier, BertTextEncoder,
                           load_tokenizer)
from .contrastive import DualEncoder, build_similarity_matrix, symmetric_info_nce
from .datasets import (MusicGraphDataset, TextTagDataset, alternating_loader,
                       collate_texts, make_loader)
from .fusion_model import (GNNBertFusion, masked_genre_loss,
                           masked_multitask_loss)
from .gnn_model import GNNClassifier, GNNEncoder
from .splits import apply_text_source, assert_no_leakage
from pathlib import Path as _Path  # noqa: E402

from .utils import (
    SYNTHETIC,
    atomic_torch_save,
    autocast_ctx,
    detect_provenance,
    count_parameters,
    ensure_dir,
    get_device,
    get_logger,
    load_config,
    log_vram,
    make_grad_scaler,
    parse_overrides,
    REAL,
    reset_vram_peak,
    resolve_path,
    save_json,
    set_seed,
    vocabulary_hash,
)

LOGGER = get_logger("gbmc.train")

# Defaults; config.data.{tag,emotion,caption}_corpora override them per run.
TAG_DATASETS = ("mtat",)                    # corpora that carry the tag vocabulary
EMOTION_DATASETS = ("deam",)                # corpora that carry valence/arousal
CAPTION_DATASETS = ("musiccaps",)           # corpora with free-text captions
GENRE_DATASETS = ("fma",)                   # the only corpus with a genre label


def target_freeze_mode(cfg) -> str:
    """The freeze mode a run should end up training in.

    ``bert.freeze_mode`` is the setting that decides this. It used to be ignored
    entirely -- the mode was derived from ``freeze_epochs`` alone -- so a sweep
    asking for `frozen_probe` with `freeze_epochs=0` silently trained `full_ft`,
    and two of its three "freeze modes" were the same configuration.
    """
    mode = str(cfg.get("bert", {}).get("freeze_mode", "top_n"))
    if mode not in FREEZE_MODES:
        raise ValueError(f"bert.freeze_mode must be one of {FREEZE_MODES}, got {mode!r}")
    return mode


def starting_freeze_mode(cfg) -> str:
    """What the encoder starts as, honouring the warm-up schedule.

    ``freeze_epochs > 0`` means: keep the encoder frozen for that many epochs,
    then switch to :func:`target_freeze_mode`. With ``freeze_epochs == 0`` the
    target mode applies from step one.
    """
    if int(cfg.get("bert", {}).get("freeze_epochs", 0)) > 0:
        return "frozen_probe"
    return target_freeze_mode(cfg)


def corpora_for(cfg, kind: str) -> tuple:
    """Which corpora a task should draw from, per config.

    Mixing corpora with different label vocabularies is not free: MusicCaps
    aspects barely intersect the MTAT top-50, so pooling them hands most
    MusicCaps rows an all-negative label vector and drags macro-F1 down for a
    reason that has nothing to do with the model.
    """
    defaults = {"tag": TAG_DATASETS, "emotion": EMOTION_DATASETS,
                "caption": CAPTION_DATASETS, "genre": GENRE_DATASETS}
    value = cfg.get("data", {}).get(f"{kind}_corpora") if cfg else None
    return tuple(value) if value else defaults[kind]


# --------------------------------------------------------------------------- #
# data plumbing
# --------------------------------------------------------------------------- #
class DataBundle:
    """Everything a task needs, resolved once from config + flags."""

    def __init__(self, cfg, synthetic: bool):
        self.cfg = cfg
        self.synthetic = synthetic
        if synthetic:
            root = resolve_path(cfg["synthetic"]["out_dir"])
            manifest_path = root / "manifest.csv"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"{manifest_path} not found -- run `python -m src.synthetic` first"
                )
            self.manifest = pd.read_csv(manifest_path)
            with open(root / "tags.json", "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.tag_vocab = list(payload["tags"])
            self.genres = list(payload.get("genres", []))
            self.graph_dir = root / "graphs"
            self.h5_path = None
            self.mel_h5 = root / "mels.h5"
            self.norm_stats = None      # synthetic graphs are already normalised
        else:
            self.manifest, self.tag_vocab, self.genres = _load_real_manifests(cfg)
            self.graph_dir = None
            processed = resolve_path(cfg["paths"]["processed"])
            corpora = sorted({str(d) for d in self.manifest.get("dataset", [])})

            # one cache per corpus; fall back to the shared file if that is what
            # exists, so an older extraction still loads
            self.h5_path, self.mel_h5, self.norm_stats = {}, {}, {}
            for name in corpora:
                features = processed / f"features_{name}.h5"
                mels = processed / f"mels_{name}.h5"
                stats = processed / f"norm_stats_{name}.json"
                self.h5_path[name] = features if features.exists() else processed / "features.h5"
                self.mel_h5[name] = mels if mels.exists() else processed / "mels.h5"
                stats_path = stats if stats.exists() else processed / "norm_stats.json"
                if stats_path.exists():
                    payload = json.loads(stats_path.read_text(encoding="utf-8"))
                    if payload.get("split") != "train":
                        raise RuntimeError(
                            f"{stats_path} was not computed on the train split; "
                            "refusing to run"
                        )
                    self.norm_stats[name] = payload
            missing = [n for n in corpora if not _Path(self.h5_path[n]).exists()]
            if missing:
                LOGGER.warning("no feature cache for %s -- those rows will fail to load",
                               missing)

        # swap `text` for the configured Xtext variant before anything
        # tokenises it. Doing it here means every task, every script and every
        # notebook sees the same choice.
        if not synthetic:
            self.manifest = apply_text_source(self.manifest, cfg)
        self.text_source = str(cfg.get("data", {}).get("text_source", "caption_masked"))

        # train-only valence/arousal statistics, cached beside the splits
        # and injected into cfg so the loss and the metrics both see them.
        self.emotion_stats = self._emotion_stats()
        if self.emotion_stats:
            cfg.setdefault("multitask", {})["emotion_stats"] = self.emotion_stats

        # the assertion the whole pipeline hangs on
        assert_no_leakage(self.manifest)
        # Recorded so every checkpoint and result JSON carries its own
        # provenance; evaluate.py refuses to build a report from synthetic ones.
        self.provenance = SYNTHETIC if synthetic else detect_provenance(self.manifest)

    def _emotion_stats(self) -> dict:
        """Train-split valence/arousal statistics, computed once and cached.

        Cached to disk rather than recomputed per run so that every run in a
        seed sweep standardises against exactly the same numbers -- a per-run
        recompute would be identical here, but only because the split is fixed,
        and relying on that is how a subtle inconsistency gets in later.
        """
        from .splits import compute_emotion_stats

        emotion = corpora_for(self.cfg, "emotion")
        rows = self.manifest[self.manifest["dataset"].isin(list(emotion))]
        if not len(rows):
            return {}

        path = resolve_path(self.cfg["paths"]["splits"]) / "emotion_stats.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("split") != "train":
                raise RuntimeError(
                    f"{path} was not computed on the train split; refusing to run"
                )
            return payload

        stats = compute_emotion_stats(rows, split="train")
        save_json(stats, path)
        return stats

    def text_dataset(self, split: str, datasets=None) -> TextTagDataset:
        """Text + labels only, for the tasks that never touch audio.

        Keeps Task 1 runnable from the manifests alone, which is what makes the
        Kaggle payload 3 MB instead of gigabytes.
        """
        return TextTagDataset(
            self.manifest,
            cfg=self.cfg,
            tag_vocab=self.tag_vocab,
            split=split,
            datasets=datasets,
        )

    def dataset(self, split: str, datasets=None) -> MusicGraphDataset:
        return MusicGraphDataset(
            self.manifest,
            cfg=self.cfg,
            h5_path=self.h5_path,
            graph_dir=self.graph_dir,
            tag_vocab=self.tag_vocab,
            norm_stats=self.norm_stats,
            split=split,
            datasets=datasets,
        )

    def frame(self, split: str, datasets=None) -> pd.DataFrame:
        frame = self.manifest[self.manifest["split"] == split]
        if datasets is not None:
            frame = frame[frame["dataset"].isin(list(datasets))]
        return frame.reset_index(drop=True)


def _load_real_manifests(cfg):
    """Concatenate whatever per-dataset manifests exist under ``paths.splits``."""
    splits_dir = resolve_path(cfg["paths"]["splits"])
    frames = []
    for name in ("mtat", "fma", "deam", "musiccaps"):
        path = splits_dir / f"{name}_manifest.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["dataset"] = frame.get("dataset", name)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"no manifests in {splits_dir} -- run `python -m src.splits` "
            "(or `make verify-data`) first, or pass --synthetic"
        )
    manifest = pd.concat(frames, ignore_index=True)

    source = str(cfg.get("tags", {}).get("vocab_source", "mtat"))
    vocab_path = (splits_dir / "musiccaps_tag_vocab.json" if source == "musiccaps"
                  else splits_dir / "tag_vocab.json")
    if not vocab_path.exists():
        LOGGER.warning("no vocabulary at %s; falling back to tag_vocab.json", vocab_path)
        vocab_path = splits_dir / "tag_vocab.json"
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        tag_vocab = list(payload["tags"] if isinstance(payload, dict) else payload)
        LOGGER.info("tag vocabulary: %d tags from %s (tags.vocab_source=%s)",
                    len(tag_vocab), vocab_path.name, source)
    else:
        pool: set[str] = set()
        for value in manifest.get("y_tags", []):
            try:
                pool.update(json.loads(value) if isinstance(value, str) else (value or []))
            except (TypeError, ValueError):
                continue
        tag_vocab = sorted(pool)[: int(cfg["tags"]["top_k"])]
    return manifest, tag_vocab, []


# --------------------------------------------------------------------------- #
# optimisation helpers
# --------------------------------------------------------------------------- #
def build_optimizer(model, cfg):
    """AdamW over the model's own parameter groups (BERT lr != head lr)."""
    train_cfg = cfg["train"]
    groups = model.param_groups(
        float(train_cfg["lr_bert"]), float(train_cfg["lr_head"]),
        float(train_cfg["weight_decay"]),
    )
    if not groups:
        raise RuntimeError("no trainable parameters -- check the freeze settings")
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg, steps_per_epoch: int, epochs: int):
    """Linear warmup then linear decay, on optimiser steps not batches."""
    total = max(1, int(steps_per_epoch * max(epochs, 1)))
    warmup = max(1, int(total * float(cfg["train"].get("warmup_ratio", 0.1))))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EarlyStopper:
    """Patience on a maximised metric, with the best state kept in memory."""

    def __init__(self, patience: int = 3, mode: str = "max"):
        self.patience = int(patience)
        self.mode = mode
        self.best = -math.inf if mode == "max" else math.inf
        self.best_epoch = -1
        self.bad_epochs = 0

    def step(self, value: float, epoch: int) -> bool:
        """Return True when this epoch is the new best."""
        if value is None or not np.isfinite(value):
            self.bad_epochs += 1
            return False
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best, self.best_epoch, self.bad_epochs = float(value), int(epoch), 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def save_checkpoint(model, path, epoch: int, metrics: dict, cfg, extra: dict | None = None):
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "metrics": metrics,
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg),
        # travels with the weights, so a checkpoint can never lose its origin
        "provenance": (extra or {}).get("provenance", REAL),
    }
    payload.update(extra or {})
    # atomic: a checkpoint killed mid-write is unloadable, and an unloadable
    # "latest" checkpoint costs the whole run
    return atomic_torch_save(payload, path)


class _Tracker:
    """Thin wandb / tensorboard shim so the loops stay readable."""

    def __init__(self, cfg, task: int, seed: int, enabled: bool = True):
        self.wandb = None
        self.tb = None
        if not enabled:
            return
        train_cfg = cfg["train"]
        if bool(train_cfg.get("wandb", False)):
            try:
                import wandb

                self.wandb = wandb
                wandb.init(project=str(train_cfg.get("wandb_project", "gnn-bert-music-context")),
                           name=f"task{task}_seed{seed}",
                           config=cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg))
            except Exception as exc:  # pragma: no cover - optional dependency
                LOGGER.warning("wandb unavailable (%s); continuing without it", exc)
        if bool(train_cfg.get("tensorboard", False)):
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(str(ensure_dir(f"runs/task{task}_seed{seed}")))
            except Exception as exc:  # pragma: no cover - optional dependency
                LOGGER.warning("tensorboard unavailable (%s); continuing without it", exc)

    def log(self, payload: dict, step: int) -> None:
        if self.wandb is not None:
            self.wandb.log(payload, step=step)
        if self.tb is not None:
            for key, value in payload.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    self.tb.add_scalar(key, value, step)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        if self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception:  # pragma: no cover
                pass


# --------------------------------------------------------------------------- #
# shared evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_scores(model, loader, device, cfg, task: int, tokenizer=None):
    """Run inference and gather scores/targets/embeddings for a whole split."""
    model.eval()
    amp = bool(cfg.get("amp", True))
    max_length = int(cfg["bert"]["max_length"])

    scores, targets, valence_p, arousal_p, valence_t, arousal_t = [], [], [], [], [], []
    embeddings, genres, track_ids, texts, genre_logits = [], [], [], [], []

    for batch in loader:
        batch = batch.to(device)
        ids = mask = None
        if task in (1, 3) and tokenizer is not None:
            ids, mask = collate_texts(batch, tokenizer, max_length, device)
        with autocast_ctx(amp, device.type):
            if task == 1:
                logits = model(ids, mask)
                out = {"tag_logits": logits, "z": None}
            elif task == 2:
                out = model(batch)
            else:
                out = model(batch, ids, mask)

        if out.get("tag_logits") is not None:
            scores.append(torch.sigmoid(out["tag_logits"].float()).cpu().numpy())
            targets.append(_batch_tags(batch, out["tag_logits"].shape).cpu().numpy())
        if out.get("genre_logits") is not None:
            genre_logits.append(out["genre_logits"].float().cpu().numpy())
        if out.get("valence") is not None:
            valence_p.append(out["valence"].float().cpu().numpy())
            arousal_p.append(out["arousal"].float().cpu().numpy())
            valence_t.append(batch.y_valence.float().cpu().numpy())
            arousal_t.append(batch.y_arousal.float().cpu().numpy())
        if out.get("z") is not None:
            embeddings.append(out["z"].float().cpu().numpy())
        genres.append(_batch_genre(batch).cpu().numpy())
        track_ids += list(batch.track_id) if isinstance(batch.track_id, list) else [batch.track_id]
        texts += list(batch.text) if isinstance(batch.text, list) else [batch.text]

    def _cat(chunks):
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0,))

    return {
        "scores": _cat(scores),
        "targets": _cat(targets),
        "genre_logits": _cat(genre_logits) if genre_logits else None,
        "valence_pred": _cat(valence_p),
        "arousal_pred": _cat(arousal_p),
        "valence_true": _cat(valence_t),
        "arousal_true": _cat(arousal_t),
        "embeddings": _cat(embeddings),
        "genres": _cat(genres),
        "track_ids": track_ids,
        "texts": texts,
    }


def _batch_tags(batch, logits_shape) -> torch.Tensor:
    y = getattr(batch, "y_tags", None)
    if y is None:
        return torch.full(logits_shape, -1.0)
    y = y.float()
    return y.view(logits_shape[0], -1) if y.dim() == 1 else y


def _batch_genre(batch) -> torch.Tensor:
    for field in ("true_genre", "y_genre"):
        value = getattr(batch, field, None)
        if value is not None:
            return value.view(-1).long()
    return torch.full((1,), -1, dtype=torch.long)


def tagging_metrics(collected: dict, thresholds=None, prefix: str = "") -> dict:
    """Macro/micro F1 + AUC-PR. Never accuracy -- see src/metrics.py."""
    y_true, y_score = collected["targets"], collected["scores"]
    if y_true.size == 0:
        return {f"{prefix}macro_f1": float("nan"), f"{prefix}micro_f1": float("nan")}
    return {
        f"{prefix}macro_f1": M.macro_f1(y_true, y_score, thresholds),
        f"{prefix}micro_f1": M.micro_f1(y_true, y_score, thresholds),
        f"{prefix}mean_auc_pr": M.mean_auc_pr(y_true, y_score),
        f"{prefix}n_rows": int(y_true.shape[0]),
    }


def emotion_metrics(collected: dict, prefix: str = "", cfg=None) -> dict:
    """MAE / RMSE / R2 on the **original 1-9 scale**.

    The model is trained on standardised targets (B1.2), so its raw output is in
    standard-deviation units. An MAE of 0.8 sd means nothing to a reader who
    knows DEAM is a 1-9 scale, so predictions are inverted here before scoring.
    R2 is invariant to the affine transform and reads the same either way; MAE
    and RMSE are not, which is exactly why this cannot be skipped.
    """
    if collected["valence_true"].size == 0:
        return {}
    mt = (cfg or {}).get("multitask", {}) or {}
    stats = (mt.get("emotion_stats") or {}) if mt.get("standardise_targets", True) else {}

    def unscale(values, name):
        scale = stats.get(name)
        if not scale:
            return values
        return values * float(scale["std"]) + float(scale["mean"])

    out = M.regression_metrics(collected["valence_true"],
                               unscale(collected["valence_pred"], "valence"),
                               prefix=f"{prefix}valence_")
    out.update(M.regression_metrics(collected["arousal_true"],
                                    unscale(collected["arousal_pred"], "arousal"),
                                    prefix=f"{prefix}arousal_"))
    out[f"{prefix}emotion_target_scale"] = "raw 1-9" if not stats else "standardised"
    return out


# --------------------------------------------------------------------------- #
# task 1 -- BERT tag classifier (also baseline B3)
# --------------------------------------------------------------------------- #
def run_task1(cfg, args, bundle: DataBundle, device) -> dict:
    tokenizer = load_tokenizer(cfg["bert"]["model_name"])
    model = BertTagClassifier(
        n_tags=len(bundle.tag_vocab),
        model_name=cfg["bert"]["model_name"],
        freeze_mode=starting_freeze_mode(cfg),
        unfreeze_top_n=int(cfg["bert"]["unfreeze_top_n_layers"]),
        gradient_checkpointing=bool(cfg["bert"].get("gradient_checkpointing", False)),
    ).to(device)

    # Task 1 is text-only: no audio, no feature cache, no graphs.
    loaders = {
        split: make_loader(bundle.text_dataset(split, corpora_for(cfg, 'tag')), cfg,
                           shuffle=(split == "train"), seed=args.seed,
                           num_workers=args.num_workers)
        for split in ("train", "val", "test")
    }

    def step(batch):
        ids, mask = collate_texts(batch, tokenizer, int(cfg["bert"]["max_length"]), device)
        logits = model(ids, mask)
        return masked_multitask_loss({"tag_logits": logits, "valence": None, "arousal": None},
                                     batch, cfg)

    return _fit(model, loaders, cfg, args, device, task=1, step_fn=step,
                tokenizer=tokenizer, tag_vocab=bundle.tag_vocab,
                provenance=bundle.provenance,
                text_source=bundle.text_source)


# --------------------------------------------------------------------------- #
# task 2 -- GNN on audio graphs
# --------------------------------------------------------------------------- #
def task2_target(cfg) -> str:
    """``genre`` (FMA-small, 8-way single label) or ``tags`` (MTAT multi-label).

    The PDF deliverable for Task 2 is genre classification on FMA-small, so that
    is the headline. MTAT multi-label tagging is kept as a second domain rather
    than dropped: it is the only corpus here whose label space the fusion tasks
    also use, so it is what makes Task 2 and Task 3 comparable.
    """
    target = str(cfg.get("data", {}).get("task2_target", "genre")).lower()
    if target not in ("genre", "tags"):
        raise ValueError(f"data.task2_target must be 'genre' or 'tags', got {target!r}")
    return target


def genre_names(cfg, manifest) -> list:
    """Ordered FMA-small genre names, index i == ``y_genre == i``."""
    configured = cfg.get("data", {}).get("genre_names")
    if configured:
        return list(configured)
    labels = sorted({int(v) for v in manifest.get("y_genre", []) if int(v) >= 0})
    return [f"genre_{i}" for i in labels]


def run_task2(cfg, args, bundle: DataBundle, device) -> dict:
    gnn_cfg = cfg["gnn"]
    target = task2_target(cfg)
    corpora = (corpora_for(cfg, "genre") if target == "genre"
               else corpora_for(cfg, "tag"))

    if target == "genre":
        frame = bundle.manifest[bundle.manifest["dataset"].isin(corpora)]
        names = genre_names(cfg, frame)
        n_genres = len(names)
        if n_genres < 2:
            raise RuntimeError(
                f"task2_target=genre but {corpora} carries {n_genres} genre "
                "label(s); check y_genre in the manifest"
            )
        LOGGER.info("task 2 target=genre: %d classes over %s (%d clips)",
                    n_genres, ",".join(corpora), len(frame))
    else:
        names, n_genres = [], 0

    model = GNNClassifier(
        # genre-only builds no tag head, so trainable_params describes the model
        # that is actually trained
        n_tags=0 if target == "genre" else len(bundle.tag_vocab),
        n_genres=n_genres,
        in_dim=int(cfg["graph"]["node_feat_dim"]),
        hidden_dim=int(gnn_cfg["hidden_dim"]),
        num_layers=int(gnn_cfg["num_layers"]),
        conv=str(gnn_cfg["conv"]),
        dropout=float(gnn_cfg["dropout"]),
        readout=str(gnn_cfg["readout"]),
        residual=bool(gnn_cfg["residual"]),
        heads=int(gnn_cfg.get("heads", 4)),
    ).to(device)

    loaders = {
        split: make_loader(bundle.dataset(split, corpora), cfg,
                           shuffle=(split == "train"), seed=args.seed,
                           num_workers=args.num_workers)
        for split in ("train", "val", "test")
    }

    def step(batch):
        out = model(batch)
        if target == "genre":
            return masked_genre_loss(out, batch, cfg)
        out.setdefault("valence", None)
        out.setdefault("arousal", None)
        return masked_multitask_loss(out, batch, cfg)

    return _fit(model, loaders, cfg, args, device, task=2, step_fn=step,
                tokenizer=None,
                tag_vocab=names if target == "genre" else bundle.tag_vocab,
                early_stop_metric="genre_macro_f1" if target == "genre" else None,
                extra_result={"task2_target": target, "corpora": list(corpora),
                              "genre_names": names},
                provenance=bundle.provenance,
                text_source=bundle.text_source)


# --------------------------------------------------------------------------- #
# task 3 -- fusion + multi-task emotion
# --------------------------------------------------------------------------- #
def run_task3(cfg, args, bundle: DataBundle, device) -> dict:
    gnn_cfg, fusion_cfg = cfg["gnn"], cfg["fusion"]
    tokenizer = load_tokenizer(cfg["bert"]["model_name"])
    gnn = GNNEncoder(
        in_dim=int(cfg["graph"]["node_feat_dim"]), hidden_dim=int(gnn_cfg["hidden_dim"]),
        num_layers=int(gnn_cfg["num_layers"]), conv=str(gnn_cfg["conv"]),
        dropout=float(gnn_cfg["dropout"]), readout=str(gnn_cfg["readout"]),
        residual=bool(gnn_cfg["residual"]), heads=int(gnn_cfg.get("heads", 4)),
    )
    bert = BertTextEncoder(
        cfg["bert"]["model_name"],
        freeze_mode=starting_freeze_mode(cfg),
        unfreeze_top_n=int(cfg["bert"]["unfreeze_top_n_layers"]),
        gradient_checkpointing=bool(cfg["bert"].get("gradient_checkpointing", False)),
    )
    model = GNNBertFusion(
        gnn, bert, mode=str(fusion_cfg["mode"]), shared_dim=int(fusion_cfg["shared_dim"]),
        n_heads=int(fusion_cfg["n_heads"]), n_tags=len(bundle.tag_vocab),
        predict_emotion=True,
    ).to(device)

    # Task 3 alternates tag-bearing and emotion-bearing batches so both heads
    # get gradient inside every optimiser window.
    tag_loader = make_loader(bundle.dataset("train", corpora_for(cfg, "tag")), cfg, shuffle=True,
                             seed=args.seed, num_workers=args.num_workers)
    emo_loader = make_loader(bundle.dataset("train", corpora_for(cfg, "emotion")), cfg, shuffle=True,
                             seed=args.seed + 1, num_workers=args.num_workers)
    ratio = tuple(cfg.get("multitask", {}).get("batch_ratio", (4, 1)))
    LOGGER.info("task 3 alternating %d tag : %d emotion batches (DEAM is ~1:14 "
                "the size of MTAT)", ratio[0], ratio[1])
    # Evaluation must see the same corpora training did. Leaving this unfiltered
    # scored an MTAT run over 7,079 test rows instead of 3,775: MusicCaps is in
    # the {mtat, musiccaps} set that datasets.py gives real label vectors, so its
    # rows arrived as confident zeros against the MTAT vocabulary and counted as
    # true negatives on every tag. That is the exact failure corpora_for's own
    # docstring warns about, and it cost ~0.07 macro-F1 on every Task 3 row.
    eval_corpora = tuple(dict.fromkeys(
        tuple(corpora_for(cfg, "tag")) + tuple(corpora_for(cfg, "emotion"))))
    LOGGER.info("task 3 evaluating on %s", ", ".join(eval_corpora))
    loaders = {
        "train": _AlternatingTrainLoader(tag_loader, emo_loader, ratio=ratio),
        "val": make_loader(bundle.dataset("val", eval_corpora), cfg, shuffle=False,
                           seed=args.seed, num_workers=args.num_workers),
        "test": make_loader(bundle.dataset("test", eval_corpora), cfg, shuffle=False,
                            seed=args.seed, num_workers=args.num_workers),
    }

    max_length = int(cfg["bert"]["max_length"])

    def step(batch):
        ids, mask = collate_texts(batch, tokenizer, max_length, device)
        out = model(batch, ids, mask)
        return masked_multitask_loss(out, batch, cfg)

    return _fit(model, loaders, cfg, args, device, task=3, step_fn=step,
                tokenizer=tokenizer, tag_vocab=bundle.tag_vocab,
                # emotion is auxiliary, so selection follows the tagging
                # metric. Early-stopping on a blended score would let a
                # collapsing tag head hide behind a good regression fit.
                early_stop_metric="macro_f1",
                extra_result={"batch_ratio": list(ratio),
                              "fusion_mode": str(fusion_cfg["mode"]),
                              "emotion_stats": bundle.emotion_stats},
                provenance=bundle.provenance,
                text_source=bundle.text_source)


class _AlternatingTrainLoader:
    """Wraps :func:`alternating_loader` so it can be re-iterated per epoch.

    The ratio is not decoration. DEAM has ~1,200 training tracks against MTAT's
    16,881, roughly 1:14. Alternating one-for-one would show the emotion head
    every DEAM track fourteen times per MTAT epoch, which overfits it hard while
    the tag head sees each example once. ``multitask.batch_ratio`` sets how many
    tag batches pass between emotion batches; the value used is recorded in the
    result payload so the table can state it.
    """

    def __init__(self, loader_a, loader_b, ratio=(4, 1)):
        self.loader_a = loader_a
        self.loader_b = loader_b
        self.ratio = tuple(int(v) for v in ratio)

    def __iter__(self):
        for batch, _tag in alternating_loader(self.loader_a, self.loader_b,
                                              ratio=self.ratio):
            yield batch

    def __len__(self):
        n_a, n_b = self.ratio
        # the emotion loader is cycled, so length follows the tag loader
        return len(self.loader_a) + max(1, len(self.loader_a) * n_b // max(n_a, 1))


# --------------------------------------------------------------------------- #
# task 4 -- contrastive retrieval
# --------------------------------------------------------------------------- #
def run_task4(cfg, args, bundle: DataBundle, device) -> dict:
    gnn_cfg, con_cfg = cfg["gnn"], cfg["contrastive"]
    tokenizer = load_tokenizer(cfg["bert"]["model_name"])
    gnn = GNNEncoder(
        in_dim=int(cfg["graph"]["node_feat_dim"]), hidden_dim=int(gnn_cfg["hidden_dim"]),
        num_layers=int(gnn_cfg["num_layers"]), conv=str(gnn_cfg["conv"]),
        dropout=float(gnn_cfg["dropout"]), readout=str(gnn_cfg["readout"]),
        residual=bool(gnn_cfg["residual"]), heads=int(gnn_cfg.get("heads", 4)),
    )
    bert = BertTextEncoder(cfg["bert"]["model_name"], freeze_mode="top_n",
                           unfreeze_top_n=int(cfg["bert"]["unfreeze_top_n_layers"]))
    model = DualEncoder(
        gnn, bert, shared_dim=int(cfg["fusion"]["shared_dim"]),
        temperature_init=float(con_cfg["temperature_init"]),
        learnable=bool(con_cfg["learnable_temperature"]),
        clamp_logit_scale=float(con_cfg["clamp_logit_scale"]),
    ).to(device)

    precompute = bool(args.precompute_text_embeddings or con_cfg.get("precompute_text", False))
    datasets = {
        split: bundle.dataset(split, corpora_for(cfg, "caption"))
        for split in ("train", "val", "test")
    }
    # contrastive wants the biggest batch that fits; precomputed text is what
    # makes cfg.contrastive.batch_size reachable on 4 GB
    batch_size = int(con_cfg["batch_size"]) if precompute else int(cfg["train"]["batch_size"])
    batch_size = max(2, min(batch_size, max(len(datasets["train"]), 2)))

    loaders = {
        split: make_loader(datasets[split], cfg, shuffle=(split == "train"), seed=args.seed,
                           batch_size=batch_size,
                           drop_last=(split == "train" and len(datasets[split]) > batch_size),
                           num_workers=args.num_workers)
        for split in ("train", "val", "test")
    }

    text_index: dict[str, int] = {}
    if precompute:
        texts = [str(t) for t in bundle.manifest["text"].fillna("music").tolist()]
        ids = [str(t) for t in bundle.manifest["track_id"].tolist()]
        text_index = {track: i for i, track in enumerate(ids)}
        model.set_precomputed_text(
            bert.precompute_embeddings(texts, tokenizer=tokenizer, device=device,
                                       max_length=int(cfg["bert"]["max_length"]))
        )

    max_length = int(cfg["bert"]["max_length"])

    def encode(batch):
        g = model.encode_graph(batch)
        if precompute:
            track_ids = batch.track_id if isinstance(batch.track_id, list) else [batch.track_id]
            idx = [text_index.get(str(t), 0) for t in track_ids]
            t = model.encode_text_precomputed(idx)
        else:
            ids_, mask_ = collate_texts(batch, tokenizer, max_length, device)
            t = model.encode_text(ids_, mask_)
        return g, t

    def step(batch):
        g, t = encode(batch)
        loss = symmetric_info_nce(g, t, model.get_logit_scale())
        return loss, {"loss_total": float(loss.detach().item()),
                      "temperature": model.temperature, "batch": int(g.shape[0])}

    @torch.no_grad()
    def evaluate(loader) -> tuple[dict, np.ndarray, list]:
        model.eval()
        graphs, texts_emb, track_ids, captions = [], [], [], []
        for batch in loader:
            batch = batch.to(device)
            with autocast_ctx(bool(cfg.get("amp", True)), device.type):
                g, t = encode(batch)
            graphs.append(g.float().cpu())
            texts_emb.append(t.float().cpu())
            track_ids += list(batch.track_id) if isinstance(batch.track_id, list) else [batch.track_id]
            captions += list(batch.text) if isinstance(batch.text, list) else [batch.text]
        if not graphs:
            return {}, np.zeros((0, 0)), []
        sim = build_similarity_matrix(torch.cat(graphs), torch.cat(texts_emb))
        return M.retrieval_metrics(sim), sim, list(zip(track_ids, captions))

    return _fit(model, loaders, cfg, args, device, task=4, step_fn=step, tokenizer=tokenizer,
                tag_vocab=bundle.tag_vocab, eval_fn=evaluate,
                early_stop_metric="mean_R@10", provenance=bundle.provenance,
                text_source=bundle.text_source,
                extra_result={
                    "precompute_text": precompute, "contrastive_batch_size": batch_size,
                })


# --------------------------------------------------------------------------- #
# the shared training loop
# --------------------------------------------------------------------------- #
def _fit(model, loaders, cfg, args, device, task: int, step_fn, tokenizer,
         tag_vocab, eval_fn=None, early_stop_metric: str | None = None,
         extra_result: dict | None = None, provenance: str = REAL,
         text_source: str = "caption_masked") -> dict:
    """Train, validate, early-stop, then evaluate test once with frozen thresholds."""
    train_cfg = cfg["train"]
    epochs = int(train_cfg["epochs"])
    accum = max(1, int(train_cfg["grad_accum_steps"]))
    amp = bool(cfg.get("amp", True))
    clip = float(train_cfg.get("grad_clip", 1.0))
    freeze_epochs = int(cfg["bert"].get("freeze_epochs", 0))
    unfreeze_top_n = int(cfg["bert"].get("unfreeze_top_n_layers", 4))

    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = max(1, math.ceil(_loader_len(loaders["train"]) / accum))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch, epochs)
    scaler = make_grad_scaler(amp, device.type)
    tracker = _Tracker(cfg, task, args.seed, enabled=not args.dry_run)

    metric_key = early_stop_metric or str(train_cfg.get("early_stop_metric", "val_macro_f1"))
    metric_key = metric_key.replace("val_", "")
    stopper = EarlyStopper(int(train_cfg.get("early_stop_patience", 3)))
    ckpt_dir = ensure_dir(cfg["paths"].get("checkpoints", "results/checkpoints"))
    # Checkpoints carry the run tag for the same reason results do. Without it
    # the seven fusion modes of the ablation each overwrite the last, and the
    # case studies -- which need one specific MusicCaps model -- would silently
    # load whichever run happened to finish most recently.
    suffix = f"_{args.run_tag}" if getattr(args, "run_tag", None) else ""
    best_state = None
    best_thresholds = None
    history: list[dict] = []
    reset_vram_peak()

    for epoch in range(1, epochs + 1):
        # ---- BERT unfreeze transition ------------------------------------ #
        text_encoder = model.bert if isinstance(getattr(model, "bert", None), BertTextEncoder) else \
            getattr(model, "encoder", None)
        if isinstance(text_encoder, BertTextEncoder) and freeze_epochs and epoch == freeze_epochs + 1:
            # switch to whatever bert.freeze_mode actually asked for, not always top_n
            target = target_freeze_mode(cfg)
            if target == "top_n":
                text_encoder.unfreeze_top_layers(unfreeze_top_n)
            else:
                text_encoder.set_freeze_mode(target)
            # the optimiser must be rebuilt: parameters that were frozen at
            # construction time are absent from its param groups
            optimizer = build_optimizer(model, cfg)
            scheduler = build_scheduler(optimizer, cfg, steps_per_epoch,
                                        max(1, epochs - epoch + 1))
            LOGGER.info("epoch %d: unfroze top %d BERT layers and rebuilt the optimiser",
                        epoch, unfreeze_top_n)

        model.train()
        epoch_start = time.time()
        running: dict[str, float] = {}
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(loaders["train"]):
            batch = batch.to(device)
            with autocast_ctx(amp, device.type):
                loss, parts = step_fn(batch)
            if not torch.isfinite(loss):
                LOGGER.warning("non-finite loss at epoch %d step %d; skipping batch", epoch, i)
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0:
                if clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            n_batches += 1
            for key, value in parts.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    running[key] = running.get(key, 0.0) + float(value)
            if args.dry_run and n_batches >= 3:
                break

        # flush a trailing partial accumulation window
        if n_batches % accum != 0:
            if clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_stats = {f"train_{k}": v / max(n_batches, 1) for k, v in running.items()}
        train_stats["train_batches"] = n_batches

        # ---- validation --------------------------------------------------- #
        if eval_fn is not None:
            val_metrics, _, _ = eval_fn(loaders["val"])
            thresholds = None
        else:
            collected = collect_scores(model, loaders["val"], device, cfg, task, tokenizer)
            # thresholds fitted on VAL, never on test
            thresholds = (
                M.tune_thresholds(collected["targets"], collected["scores"])
                if bool(cfg["eval"].get("tune_thresholds", True)) else None
            )
            val_metrics = tagging_metrics(collected, thresholds)
            val_metrics.update(emotion_metrics(collected, cfg=cfg))
            if collected.get("genre_logits") is not None:
                val_metrics.update(M.multiclass_metrics(
                    collected["genres"], collected["genre_logits"], prefix="genre_"))

        epoch_record = {"epoch": epoch, "seconds": time.time() - epoch_start}
        epoch_record.update(train_stats)
        epoch_record.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(epoch_record)
        tracker.log(epoch_record, step=epoch)
        LOGGER.info(
            "task %d epoch %d/%d | loss %.4f%s | val %s=%s | %.1fs",
            task, epoch, epochs, train_stats.get("train_loss_total", float("nan")),
            _loss_breakdown(train_stats),
            metric_key, _fmt(val_metrics.get(metric_key)), epoch_record["seconds"],
        )
        vram = log_vram(f"task{task}_epoch{epoch}")

        improved = stopper.step(val_metrics.get(metric_key, float("nan")), epoch)
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thresholds = thresholds
        if not args.dry_run:
            # `_last` is a mid-training resume point that nothing reads back;
            # tagging it made every run keep its own ~440 MB copy and filled the
            # disk during the C7 seeds. One rolling file per task/seed is enough.
            # `_best` stays tagged -- that one IS loaded, by name, and the
            # ablation would otherwise overwrite itself.
            save_checkpoint(model, ckpt_dir / f"task{task}_seed{args.seed}_last.pt",
                            epoch, epoch_record, cfg,
                            extra={"thresholds": None if thresholds is None else
                                   np.asarray(thresholds).tolist(),
                                   "provenance": provenance})
            if improved:
                save_checkpoint(model, ckpt_dir / f"task{task}_seed{args.seed}{suffix}_best.pt",
                                epoch, epoch_record, cfg,
                                extra={"thresholds": None if thresholds is None else
                                       np.asarray(thresholds).tolist(),
                                       "tag_vocab": list(tag_vocab),
        # lets a result produced on another machine be checked for
        # comparability without shipping the vocabulary alongside it
        "tag_vocab_hash": vocabulary_hash(tag_vocab),
                                       "provenance": provenance})
        if stopper.should_stop:
            LOGGER.info("early stop at epoch %d (best epoch %d, %s=%.4f)",
                        epoch, stopper.best_epoch, metric_key, stopper.best)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- test: touched once, with thresholds already frozen ---------------- #
    if eval_fn is not None:
        test_metrics, sim, pairs = eval_fn(loaders["test"])
        test_extra = {"similarity_shape": list(np.shape(sim))}
    else:
        collected = collect_scores(model, loaders["test"], device, cfg, task, tokenizer)
        test_metrics = tagging_metrics(collected, best_thresholds)
        test_metrics.update(emotion_metrics(collected, cfg=cfg))
        # the bootstrap found the val-tuned operating point unstable on a
        # 977-clip validation split, so the untuned number travels with it.
        if best_thresholds is not None and collected["targets"].size:
            test_metrics["macro_f1_fixed_half"] = M.macro_f1(
                collected["targets"], collected["scores"], 0.5)
            test_metrics["micro_f1_fixed_half"] = M.micro_f1(
                collected["targets"], collected["scores"], 0.5)
        test_extra = {
            "knn_probe": M.knn_probe(collected["embeddings"], collected["genres"],
                                     int(cfg["eval"].get("knn_probe_k", 10)))
            if collected["embeddings"].size else float("nan"),
        }
        if collected.get("genre_logits") is not None:
            test_metrics.update(M.multiclass_metrics(
                collected["genres"], collected["genre_logits"], prefix="genre_"))
        # keep the raw val/test score matrices of the *selected* model.
        # Bootstrapping the threshold tuner needs them, and re-running training
        # 100 times to get them would be absurd. The val pass is repeated here
        # rather than cached from the loop because early stopping may have
        # rolled the weights back to an earlier epoch.
        if not args.dry_run:
            val_collected = collect_scores(model, loaders["val"], device, cfg,
                                           task, tokenizer)
            _dump_scores(cfg, task, args.seed, tag_vocab, val_collected, collected,
                         run_tag=getattr(args, "run_tag", None))

    result = {
        "task": task,
        "seed": int(args.seed),
        "synthetic": bool(args.synthetic),
        "provenance": provenance,
        "text_source": text_source,
        "freeze_mode": target_freeze_mode(cfg),
        "freeze_epochs": int(cfg["bert"].get("freeze_epochs", 0)),
        "device": str(device),
        "amp": bool(amp),
        "grad_accum_steps": accum,
        "effective_batch": int(cfg["train"]["batch_size"]) * accum,
        "epochs_run": len(history),
        "best_epoch": stopper.best_epoch,
        "early_stop_metric": metric_key,
        "best_val_metric": stopper.best if np.isfinite(stopper.best) else None,
        "trainable_params": count_parameters(model),
        "n_tags": len(tag_vocab),
        "tag_vocab": list(tag_vocab),
        "thresholds": None if best_thresholds is None else np.asarray(best_thresholds).tolist(),
        "threshold_source": "val",
        # A7.2 compares models on compute, not parameter count, so both numbers
        # have to travel with every result rather than being reconstructed later
        "wall_clock_s": round(sum(float(h.get("seconds", 0.0)) for h in history), 1),
        "history": history,
        "test": test_metrics,
        "vram": vram,
    }
    result.update(test_extra)
    result.update(extra_result or {})

    result["run_tag"] = getattr(args, "run_tag", None)
    out_path = (resolve_path(cfg["paths"]["results"])
                / f"task{task}_seed{args.seed}{suffix}.json")
    if not args.dry_run:
        save_json(result, out_path)
        LOGGER.info("wrote %s", out_path)
    tracker.close()
    return result


def _dump_scores(cfg, task: int, seed: int, tag_vocab, val_collected,
                 test_collected, run_tag: str | None = None):
    """Persist val/test score matrices next to the result JSON (A7.4).

    Threshold tuning is a fit to the validation split like any other, and MTAT's
    val split is 977 clips from 14 artists. Quantifying how much the tuned
    thresholds move under resampling needs the score matrices, not the summary
    metrics, so they are written once per run and re-read by
    ``scripts/threshold_bootstrap.py``.
    """
    out_dir = ensure_dir(resolve_path(cfg["paths"]["results"]) / "scores")
    suffix = f"_{run_tag}" if run_tag else ""
    path = out_dir / f"task{task}_seed{seed}{suffix}_scores.npz"
    payload = dict(
        val_y_true=val_collected["targets"], val_y_score=val_collected["scores"],
        test_y_true=test_collected["targets"], test_y_score=test_collected["scores"],
        tag_vocab=np.asarray(list(tag_vocab), dtype=object),
    )
    for split, collected in (("val", val_collected), ("test", test_collected)):
        if collected.get("genre_logits") is not None:
            payload[f"{split}_genre_logits"] = collected["genre_logits"]
            payload[f"{split}_genre_true"] = collected["genres"]
    np.savez_compressed(path, **payload)
    LOGGER.info("wrote score matrices -> %s (val %s, test %s)", path,
                val_collected["scores"].shape, test_collected["scores"].shape)
    return path


def _loss_breakdown(train_stats: dict) -> str:
    """The per-term losses, inline in the epoch line (B1.2).

    Task 3 optimises tagging and emotion together against very different
    amounts of data. If one term collapses -- the emotion head overfitting its
    1,200 DEAM tracks, or the tag head starved of gradient -- the total loss can
    keep falling while the model quietly stops learning one of its two jobs.
    Printing the terms separately makes that visible while the run is still
    going, rather than after it finishes.
    """
    parts = []
    for key, label in (("train_loss_tags", "tag"),
                       ("train_loss_valence", "val"),
                       ("train_loss_arousal", "aro"),
                       ("train_loss_genre", "genre")):
        value = train_stats.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value) and value:
            parts.append(f"{label} {value:.3f}")
    return f" ({', '.join(parts)})" if parts else ""


def _loader_len(loader) -> int:
    try:
        return max(1, len(loader))
    except TypeError:  # pragma: no cover - generator loaders
        return 1


def _fmt(value) -> str:
    return "nan" if value is None or not np.isfinite(value) else f"{value:.4f}"


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one of the four tasks.")
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--synthetic", action="store_true",
                        help="train on the synthetic dataset from src.synthetic")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--override", nargs="*", default=[],
                        help="config overrides, e.g. --override train.epochs=1 gnn.conv=gatv2")
    parser.add_argument("--dry-run", action="store_true",
                        help="build everything and run 3 batches; write nothing")
    parser.add_argument("--precompute-text-embeddings", action="store_true",
                        help="freeze BERT and cache caption embeddings (Task 4 on 4 GB)")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--run-tag", default=None,
                        help="suffix for the result and score filenames, so two "
                             "configurations of the same task do not overwrite "
                             "each other (e.g. --run-tag fma_genre)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.override))
    if args.device:
        cfg["device"] = args.device
    args.seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    if args.num_workers is None:
        # 200 tiny synthetic graphs: worker start-up costs more than it saves
        args.num_workers = 0 if args.synthetic else int(cfg["train"].get("num_workers", 4))

    set_seed(args.seed)
    device = get_device(str(cfg.get("device", "cuda")))
    LOGGER.info("task %d | seed %d | device %s | amp %s | synthetic %s",
                args.task, args.seed, device, cfg.get("amp", True), args.synthetic)

    bundle = DataBundle(cfg, args.synthetic)
    runners = {1: run_task1, 2: run_task2, 3: run_task3, 4: run_task4}
    result = runners[args.task](cfg, args, bundle, device)

    summary = {k: v for k, v in result.items() if k not in {"history", "thresholds"}}
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
