"""Task 3: GNN + BERT fusion, all six ablation modes, and the masked loss.

The loss in this file is the single most important correctness rule in the
codebase. No track carries both MTAT tags and DEAM valence/arousal, so the label
matrix is block-structured with ``-1`` and ``nan`` holes. An unmasked BCE would
teach the model that every DEAM track has *no* tags -- 46 confident negatives
per row, thousands of rows -- and macro-F1 would collapse while the loss curve
looked healthy. :func:`masked_multitask_loss` computes each term only over the
rows that actually carry that supervision.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_logger

LOGGER = get_logger("gbmc.fusion")

__all__ = [
    "masked_genre_loss",
    "CrossAttentionFusion",
    "GatedFusion",
    "GNNBertFusion",
    "masked_multitask_loss",
    "FUSION_MODES",
]

FUSION_MODES = (
    "bert_only", "gnn_only", "early_concat", "late_concat",
    "cross_attention", "gated", "bidirectional",
)


class CrossAttentionFusion(nn.Module):
    """The graph vector attends over BERT's token states.

    ``g`` becomes a single query token and the caption's tokens are keys and
    values, so the model learns *which words* a given musical structure should
    listen to -- that alignment is what the Task 3 case studies visualise.
    Padding is masked out; without ``key_padding_mask`` the graph happily
    attends to ``[PAD]`` and the attention maps become unreadable.
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, int(n_heads), dropout=dropout,
                                          batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )

    def forward(self, g, tokens, attention_mask=None, return_attn: bool = False):
        query = self.norm_q(g).unsqueeze(1)                       # [B, 1, D]
        kv = self.norm_kv(tokens)                                 # [B, L, D]
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0                # True == ignore
        attended, weights = self.attn(
            query, kv, kv, key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=True,
        )
        fused = g + attended.squeeze(1)
        fused = fused + self.ffn(self.norm_out(fused))
        return (fused, weights.squeeze(1)) if return_attn else (fused, None)


class GatedFusion(nn.Module):
    """Learned per-dimension gate between the graph and text vectors.

    ``z = a * g + (1 - a) * t`` with ``a = sigmoid(W[g; t])``. The gate is
    diagnostic as well as functional: a gate that saturates at 0 or 1 is the
    model telling you one branch is being ignored, which is the failure the
    separate learning rates exist to prevent.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.proj = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, dim), nn.GELU()
        )

    def forward(self, g, t, return_gate: bool = False):
        alpha = self.gate(torch.cat([g, t], dim=-1))
        fused = self.proj(alpha * g + (1.0 - alpha) * t)
        return (fused, alpha) if return_gate else (fused, None)


class GNNBertFusion(nn.Module):
    """All six fusion variants behind one interface, for the ablation table.

    ``forward`` returns ``dict(tag_logits, valence, arousal, z, attn)``. ``z`` is
    the fused representation and is what the t-SNE panels and the k-NN probe are
    computed on.
    """

    def __init__(self, gnn, bert, mode: str = "cross_attention", shared_dim: int = 256,
                 n_heads: int = 4, n_tags: int = 50, predict_emotion: bool = True,
                 dropout: float = 0.1, n_genres: int = 0):
        super().__init__()
        if mode not in FUSION_MODES:
            raise ValueError(f"mode must be one of {FUSION_MODES}, got {mode!r}")
        self.mode = mode
        self.gnn = gnn
        self.bert = bert
        self.shared_dim = int(shared_dim)
        self.predict_emotion = bool(predict_emotion)
        self.n_tags = int(n_tags)

        gnn_dim = getattr(gnn, "out_dim", shared_dim)
        bert_dim = getattr(bert, "hidden_size", shared_dim)
        self.gnn_proj = nn.Linear(gnn_dim, self.shared_dim)
        self.bert_proj = nn.Linear(bert_dim, self.shared_dim)
        self.token_proj = nn.Linear(bert_dim, self.shared_dim)

        if mode == "cross_attention":
            self.cross = CrossAttentionFusion(self.shared_dim, n_heads, dropout)
            fused_dim = self.shared_dim
        elif mode == "bidirectional":
            self.cross = CrossAttentionFusion(self.shared_dim, n_heads, dropout)
            self.cross_back = CrossAttentionFusion(self.shared_dim, n_heads, dropout)
            fused_dim = self.shared_dim * 2
        elif mode == "gated":
            self.gated = GatedFusion(self.shared_dim, dropout)
            fused_dim = self.shared_dim
        elif mode in {"early_concat", "late_concat"}:
            fused_dim = self.shared_dim * 2
        else:                                    # bert_only / gnn_only
            fused_dim = self.shared_dim

        self.dropout = nn.Dropout(dropout)
        self.tag_head = nn.Sequential(
            nn.LayerNorm(fused_dim), nn.Linear(fused_dim, self.shared_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(self.shared_dim, self.n_tags),
        )
        self.emotion_head = (
            nn.Sequential(
                nn.LayerNorm(fused_dim), nn.Linear(fused_dim, self.shared_dim // 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(self.shared_dim // 2, 2),
            )
            if predict_emotion else None
        )
        self.genre_head = nn.Linear(fused_dim, int(n_genres)) if n_genres else None
        self.fused_dim = fused_dim

    # -- forward ---------------------------------------------------------- #
    def forward(self, data, input_ids=None, attention_mask=None, return_attn: bool = False):
        need_tokens = self.mode in {"cross_attention", "bidirectional"}
        attn = None

        g = t = tokens = None
        if self.mode != "bert_only":
            g_raw, _ = self.gnn(data, return_nodes=False)
            g = self.gnn_proj(g_raw)
        if self.mode != "gnn_only":
            if input_ids is None:
                raise ValueError(f"mode {self.mode!r} needs input_ids/attention_mask")
            cls, tok = self.bert(input_ids, attention_mask, return_tokens=need_tokens)
            t = self.bert_proj(cls)
            tokens = self.token_proj(tok) if tok is not None else None

        if self.mode == "gnn_only":
            z = g
        elif self.mode == "bert_only":
            z = t
        elif self.mode == "early_concat":
            # concatenate the two encoder outputs before any interaction
            z = torch.cat([g, t], dim=-1)
        elif self.mode == "late_concat":
            # same shapes, but each branch is normalised on its own first, so the
            # head cannot be dominated by whichever branch has the larger scale
            z = torch.cat([F.layer_norm(g, g.shape[-1:]), F.layer_norm(t, t.shape[-1:])], dim=-1)
        elif self.mode == "gated":
            z, attn = self.gated(g, t, return_gate=return_attn)
        elif self.mode == "cross_attention":
            z, attn = self.cross(g, tokens, attention_mask, return_attn=return_attn)
        else:                                                   # bidirectional
            g2t, attn_fwd = self.cross(g, tokens, attention_mask, return_attn=return_attn)
            t2g, _ = self.cross_back(t, g.unsqueeze(1), None, return_attn=False)
            z = torch.cat([g2t, t2g], dim=-1)
            attn = attn_fwd

        z = self.dropout(z)
        out = {"tag_logits": self.tag_head(z), "z": z, "attn": attn}
        if self.emotion_head is not None:
            emotion = self.emotion_head(z)
            out["valence"] = emotion[:, 0]
            out["arousal"] = emotion[:, 1]
        else:
            out["valence"] = None
            out["arousal"] = None
        if self.genre_head is not None:
            out["genre_logits"] = self.genre_head(z)
        return out

    # -- optimisation ----------------------------------------------------- #
    def param_groups(self, lr_bert: float, lr_head: float,
                     weight_decay: float = 0.01) -> list[dict]:
        """Pretrained BERT at ``lr_bert``; GNN, fusion and heads at ``lr_head``.

        A single shared LR reliably produces a model that ignores one branch:
        at 1e-3 the pretrained encoder is destroyed in a few hundred steps, and
        at 2e-5 the randomly initialised GNN never leaves its initialisation.
        """
        bert_params = {id(p) for p in self.bert.parameters()} if self.mode != "gnn_only" else set()
        buckets = {"bert_decay": [], "bert_no_decay": [], "head_decay": [], "head_no_decay": []}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            skip_decay = param.ndim <= 1 or name.endswith(".bias") or "LayerNorm" in name
            prefix = "bert" if id(param) in bert_params else "head"
            buckets[f"{prefix}_{'no_decay' if skip_decay else 'decay'}"].append(param)
        lrs = {"bert": lr_bert, "head": lr_head}
        groups = []
        for key, params in buckets.items():
            if not params:
                continue
            prefix = key.split("_")[0]
            groups.append({
                "params": params,
                "lr": lrs[prefix],
                "weight_decay": 0.0 if "no_decay" in key else weight_decay,
                "name": key,
            })
        return groups


# --------------------------------------------------------------------------- #
# the masked multi-task loss
# --------------------------------------------------------------------------- #
class _RunningMagnitude:
    """EMA of each term's magnitude, used to auto-balance the objectives."""

    def __init__(self, decay: float = 0.98):
        self.decay = float(decay)
        self.state: dict[str, float] = {}

    def update(self, key: str, value: float) -> float:
        if not (value == value) or value <= 0:            # nan or empty term
            return self.state.get(key, 1.0)
        prev = self.state.get(key)
        self.state[key] = value if prev is None else self.decay * prev + (1 - self.decay) * value
        return self.state[key]


_MAGNITUDES = _RunningMagnitude()


def masked_genre_loss(out: dict, batch, cfg=None):
    """Cross-entropy over the single-label genre head, ``-1`` rows masked out.

    FMA-small is one genre per clip over eight classes, so this is ordinary
    softmax cross-entropy rather than anything in the multi-label machinery
    above. It still has to mask: manifests from corpora with no genre column
    carry ``y_genre = -1``, and feeding those to ``cross_entropy`` would either
    crash on a negative index or, worse, silently train class 0.

    Returns ``(loss, parts)`` in the same shape as
    :func:`masked_multitask_loss`, so ``_fit`` needs no special case.
    """
    logits = out.get("genre_logits")
    if logits is None:
        raise ValueError(
            "genre target selected but the model exposes no genre_logits -- "
            "construct it with n_genres > 0"
        )
    device = logits.device
    y = getattr(batch, "y_genre", None)
    if y is None:
        return (torch.zeros((), device=device),
                {"loss_genre": 0.0, "n_genre": 0, "loss_total": 0.0})
    y = y.view(-1).long().to(device)
    mask = y >= 0
    count = int(mask.sum().item())
    if count == 0:
        return (torch.zeros((), device=device),
                {"loss_genre": 0.0, "n_genre": 0, "loss_total": 0.0})
    loss = F.cross_entropy(logits.float()[mask], y[mask])
    parts = {"loss_genre": float(loss.detach().item()), "n_genre": count}
    parts["loss_total"] = parts["loss_genre"]
    return loss, parts


def masked_multitask_loss(out: dict, batch, cfg, magnitudes: "_RunningMagnitude | None" = None):
    """BCE over observed tags + MSE over observed valence/arousal.

    Masking rules, both enforced here and nowhere else:

    * a tag cell counts only when ``y_tags != -1`` (and is finite)
    * a regression target counts only when it is not ``nan``

    ``multitask.auto_balance`` divides each term by an EMA of its own magnitude
    before weighting. Without it, raw MSE on DEAM's 1-9 scale sits around 4-10
    while per-tag BCE sits around 0.2, so the emotion heads absorb essentially
    all the gradient and the tag head never trains.

    Returns ``(total_loss, parts)`` where ``parts`` carries each raw term and the
    number of supervised rows behind it -- log these; a term whose ``n`` is 0 for
    a whole epoch means the alternating loader is misconfigured.
    """
    magnitudes = magnitudes if magnitudes is not None else _MAGNITUDES
    mt_cfg = cfg["multitask"] if "multitask" in cfg else {}
    alpha = float(mt_cfg.get("alpha_valence", 1.0))
    beta = float(mt_cfg.get("beta_arousal", 1.0))
    auto_balance = bool(mt_cfg.get("auto_balance", True))
    # targets are standardised with TRAIN-split statistics before the MSE.
    # Raw 1-9 valence produces squared errors of 4-10 against per-tag BCE near
    # 0.2, so without this the emotion heads take essentially all the gradient.
    emotion_stats = ((mt_cfg.get("emotion_stats") or {})
                     if mt_cfg.get("standardise_targets", True) else {})

    logits = out["tag_logits"]
    device = logits.device
    dtype = torch.float32
    parts: dict[str, float] = {}
    total = torch.zeros((), device=device, dtype=dtype)

    # ---- tags -------------------------------------------------------------
    y_tags = getattr(batch, "y_tags", None)
    tag_loss = torch.zeros((), device=device, dtype=dtype)
    n_tag_cells = 0
    if y_tags is not None and logits is not None:
        y_tags = y_tags.to(device=device, dtype=dtype)
        if y_tags.dim() == 1:
            y_tags = y_tags.view(logits.shape[0], -1)
        mask = torch.isfinite(y_tags) & (y_tags != -1.0)
        n_tag_cells = int(mask.sum().item())
        if n_tag_cells > 0:
            per_cell = F.binary_cross_entropy_with_logits(
                logits.float(), torch.clamp(y_tags, min=0.0), reduction="none"
            )
            tag_loss = (per_cell * mask.float()).sum() / mask.float().sum()
    parts["loss_tags"] = float(tag_loss.detach().item())
    parts["n_tag_cells"] = n_tag_cells

    # ---- valence / arousal ------------------------------------------------
    def _regression_term(pred, target, name: str):
        if pred is None or target is None:
            parts[f"loss_{name}"] = 0.0
            parts[f"n_{name}"] = 0
            return torch.zeros((), device=device, dtype=dtype), 0
        target = target.to(device=device, dtype=dtype).view(-1)
        pred = pred.float().view(-1)
        scale = emotion_stats.get(name)
        if scale:
            # the model predicts in standardised space, so the target moves to
            # meet it; predictions are inverted again for reporting
            target = (target - float(scale["mean"])) / max(float(scale["std"]), 1e-6)
        n = min(pred.shape[0], target.shape[0])
        pred, target = pred[:n], target[:n]
        mask = torch.isfinite(target)
        count = int(mask.sum().item())
        if count == 0:
            parts[f"loss_{name}"] = 0.0
            parts[f"n_{name}"] = 0
            return torch.zeros((), device=device, dtype=dtype), 0
        # nan targets must never reach the arithmetic: nan * 0 is still nan
        safe_target = torch.where(mask, target, torch.zeros_like(target))
        sq = (pred - safe_target) ** 2
        loss = (sq * mask.float()).sum() / mask.float().sum()
        parts[f"loss_{name}"] = float(loss.detach().item())
        parts[f"n_{name}"] = count
        return loss, count

    valence_loss, n_valence = _regression_term(
        out.get("valence"), getattr(batch, "y_valence", None), "valence"
    )
    arousal_loss, n_arousal = _regression_term(
        out.get("arousal"), getattr(batch, "y_arousal", None), "arousal"
    )

    # ---- combine ----------------------------------------------------------
    if auto_balance:
        scale_tag = magnitudes.update("tags", parts["loss_tags"]) if n_tag_cells else 1.0
        scale_val = magnitudes.update("valence", parts["loss_valence"]) if n_valence else 1.0
        scale_aro = magnitudes.update("arousal", parts["loss_arousal"]) if n_arousal else 1.0
    else:
        scale_tag = scale_val = scale_aro = 1.0
    eps = 1e-6

    if n_tag_cells:
        total = total + tag_loss / max(scale_tag, eps)
    if n_valence:
        total = total + alpha * valence_loss / max(scale_val, eps)
    if n_arousal:
        total = total + beta * arousal_loss / max(scale_aro, eps)

    parts["scale_tags"] = float(scale_tag)
    parts["scale_valence"] = float(scale_val)
    parts["scale_arousal"] = float(scale_aro)
    parts["loss_total"] = float(total.detach().item())
    return total, parts
