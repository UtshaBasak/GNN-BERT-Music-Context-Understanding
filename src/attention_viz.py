"""Attention rendering: BERT over caption tokens, GAT over graph nodes/edges.

Produces the qualitative figures the report needs:

* five Task 1 BERT attention heatmaps over caption tokens
* three Task 3 case studies pairing a caption's cross-attention with the graph
  the model was looking at

At least one case study is a **failure**, chosen automatically as the test
example with the worst per-example F1. A qualitative section made entirely of
successes is decoration; the failure is the part that tells you what the model
actually does.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .graph_builder import visualise_graph  # noqa: E402
from .utils import encoder_name_from_checkpoint, find_checkpoint, autocast_ctx, ensure_dir, get_logger, resolve_path, set_seed  # noqa: E402

LOGGER = get_logger("gbmc.attnviz")

__all__ = [
    "plot_bert_attention",
    "plot_graph_attention",
    "generate_bert_examples",
    "generate_case_studies",
]


def _clean_tokens(tokens: list[str], mask: np.ndarray | None = None) -> list[str]:
    out = []
    for i, token in enumerate(tokens):
        if mask is not None and i < len(mask) and mask[i] == 0:
            continue
        out.append(token.replace("##", "").replace("Ġ", ""))
    return out


def plot_bert_attention(tokens, attention, out_path, title: str = "",
                        layer: int = -1, max_tokens: int = 40):
    """Head-averaged CLS attention over caption tokens for one layer.

    CLS row only: the full ``L x L`` map is unreadable at 128 tokens, and the CLS
    row is the one that actually feeds the classifier head.
    """
    attn = attention[layer] if isinstance(attention, (list, tuple)) else attention
    attn = attn.detach().cpu().float().numpy()
    if attn.ndim == 4:
        attn = attn[0]                                   # first item in the batch
    cls_row = attn.mean(axis=0)[0]                       # average heads, take CLS row
    n = min(len(tokens), cls_row.shape[0], max_tokens)
    tokens, cls_row = tokens[:n], cls_row[:n]

    fig, ax = plt.subplots(figsize=(max(6, 0.32 * n), 2.6))
    image = ax.imshow(cls_row.reshape(1, -1), aspect="auto", cmap="magma")
    ax.set_yticks([])
    ax.set_xticks(range(n), tokens, rotation=75, fontsize=7, ha="right")
    ax.set_title(title or "BERT [CLS] attention over caption tokens", fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.85, label="attention")
    fig.tight_layout()
    out = resolve_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


def plot_graph_attention(data, node_scores=None, edge_alpha=None, out_path=None,
                         title: str | None = None):
    """Graph render shaded by attention.

    Node scores come from either the attention readout or, for GATv2, the
    incoming edge attention summed per node -- how much the rest of the track
    listened to that segment.
    """
    if node_scores is None and edge_alpha is not None:
        edge_index, alpha = edge_alpha
        alpha = alpha.detach().cpu().float().numpy()
        if alpha.ndim > 1:
            alpha = alpha.mean(axis=1)
        dst = edge_index[1].detach().cpu().numpy()
        node_scores = np.bincount(dst, weights=alpha, minlength=int(data.num_nodes))
    return visualise_graph(data, node_scores=node_scores, out_path=out_path, title=title)


@torch.no_grad()
def generate_bert_examples(bundle, cfg, device, seed: int, out_dir,
                           n_examples: int = 5, run_tag: str | None = None) -> list:
    """Five Task 1 attention heatmaps over real caption text."""
    from .bert_encoder import BertTagClassifier, load_tokenizer
    from .datasets import collate_texts
    from .train import TAG_DATASETS

    out = ensure_dir(out_dir)
    set_seed(seed)
    tokenizer = load_tokenizer(cfg["bert"]["model_name"])
    # eager attention is required to read the weights back out -- see
    # bert_encoder.ATTENTION_NOTE
    ckpt_probe = find_checkpoint(
        resolve_path(cfg["paths"].get("checkpoints", "results/checkpoints")),
        task=1, seed=seed, run_tag=run_tag)
    probe_payload = (torch.load(ckpt_probe, map_location=device, weights_only=False)
                     if ckpt_probe is not None else None)
    encoder_name = encoder_name_from_checkpoint(probe_payload,
                                                cfg["bert"]["model_name"])
    model = BertTagClassifier(len(bundle.tag_vocab), encoder_name,
                              freeze_mode="frozen_probe",
                              output_attentions=True).to(device)
    # Checkpoints carry their run tag, so the bare name no longer exists and
    # looking for it directly would draw attention maps from an untrained BERT
    # -- which produces a perfectly presentable, meaningless figure.
    ckpt = find_checkpoint(
        resolve_path(cfg["paths"].get("checkpoints", "results/checkpoints")),
        task=1, seed=seed, run_tag=run_tag)
    if ckpt is not None:
        from .evaluate import _load_compatible

        LOGGER.info("attention examples from %s", ckpt.name)
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        _load_compatible(model, payload["model_state"])
    else:
        LOGGER.warning("no Task 1 checkpoint for seed %s -- the attention maps "
                       "would come from an untrained encoder; skipping", seed)
        return []
    model.eval()

    dataset = bundle.dataset("test", TAG_DATASETS)
    written = []
    for i in range(min(int(n_examples), len(dataset))):
        data = dataset[i]
        text = getattr(data, "text", "") or "music"
        encoded = tokenizer([text], padding=True, truncation=True,
                            max_length=int(cfg["bert"]["max_length"]), return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits, attentions = model(encoded["input_ids"], encoded["attention_mask"],
                                   return_attn=True)
        if not attentions:
            LOGGER.warning(
                "the encoder returned no attention maps; the backbone was not "
                "loaded with attn_implementation='eager'"
            )
            break
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        mask = encoded["attention_mask"][0].cpu().numpy()
        keep = int(mask.sum())
        scores = torch.sigmoid(logits[0]).cpu().numpy()
        top = np.argsort(-scores)[:5]
        predicted = ", ".join(
            f"{bundle.tag_vocab[j]} ({scores[j]:.2f})" for j in top if j < len(bundle.tag_vocab)
        )
        path = plot_bert_attention(
            _clean_tokens(tokens[:keep]), attentions, out / f"bert_attention_{i:02d}.png",
            title=f"{getattr(data, 'track_id', '')}: {predicted}",
        )
        written.append({"track_id": getattr(data, "track_id", ""), "text": text[:200],
                        "top_tags": predicted, "plot": str(path)})
    LOGGER.info("wrote %d BERT attention examples to %s", len(written), out)
    return written


@torch.no_grad()
def generate_case_studies(bundle, cfg, device, seed: int, out_dir,
                          n_cases: int = 3, run_tag: str | None = None) -> list:
    """Three Task 3 case studies -- at least one a documented failure.

    Each case pairs the caption cross-attention (which words the graph attended
    to) with a render of the graph itself, plus the predicted and true tags, so
    the reader can see *why* a prediction went the way it did.
    """
    from .bert_encoder import BertTextEncoder, load_tokenizer
    from .datasets import collate_texts, make_loader
    from .fusion_model import GNNBertFusion
    from .gnn_model import GNNEncoder

    out = ensure_dir(out_dir)
    set_seed(seed)

    # the case studies must come from the MusicCaps fusion model.
    # MTAT's text channel is title/album/artist metadata, so a token-alignment
    # map over it shows a graph attending to an artist name -- uninformative by
    # construction. The default tag therefore names the MusicCaps run.
    #
    # The checkpoint is located and read FIRST, because the encoder it was
    # trained with decides what to build: config.yaml defaults to distilbert
    # while these runs use bert-base, and building the wrong one restores about
    # a third of the tensors and draws attention maps from the rest at random.
    ckpt = find_checkpoint(
        resolve_path(cfg["paths"].get("checkpoints", "results/checkpoints")),
        task=3, seed=seed, run_tag=run_tag or "musiccaps_cross_attention")
    if ckpt is None:
        LOGGER.warning("no Task 3 checkpoint for seed %s -- case studies from an "
                       "untrained model would be meaningless; skipping", seed)
        return []

    payload = torch.load(ckpt, map_location=device, weights_only=False)
    encoder_name = encoder_name_from_checkpoint(payload, cfg["bert"]["model_name"])
    LOGGER.info("case studies from %s (encoder %s)", ckpt.name, encoder_name)

    tokenizer = load_tokenizer(encoder_name)
    gnn = GNNEncoder(in_dim=int(cfg["graph"]["node_feat_dim"]),
                     hidden_dim=int(cfg["gnn"]["hidden_dim"]),
                     num_layers=int(cfg["gnn"]["num_layers"]),
                     conv="gatv2",           # GATv2 so per-edge attention exists
                     dropout=0.0, readout=str(cfg["gnn"]["readout"]))
    bert = BertTextEncoder(encoder_name, freeze_mode="frozen_probe")
    model = GNNBertFusion(gnn, bert, mode="cross_attention",
                          shared_dim=int(cfg["fusion"]["shared_dim"]),
                          n_heads=int(cfg["fusion"]["n_heads"]),
                          n_tags=len(bundle.tag_vocab)).to(device)

    from .evaluate import _load_compatible

    # the checkpoint's graph encoder is SAGE while this one is GATv2, so only
    # the name-and-shape compatible tensors are restored; the text tower now
    # matches exactly because it was built from the checkpoint's own config
    _load_compatible(model, payload["model_state"])
    model.eval()

    from .train import corpora_for

    # Phase C 4.3 again, and this is the half that was wrong: loading the
    # MusicCaps checkpoint is not enough if the rows still come from MTAT. The
    # first attempt produced attention maps over "8 seconds. 8 Seconds. Pain
    # Factor" -- a title, an album and an artist -- which is precisely the
    # uninformative map the instruction exists to prevent.
    dataset = bundle.dataset("test", corpora_for(bundle.cfg, "caption"))
    if len(dataset) == 0:
        LOGGER.warning("no caption-bearing test rows; skipping case studies")
        return []

    # score every example so a genuine failure can be chosen rather than assumed
    scored = []
    for i in range(len(dataset)):
        data = dataset[i]
        batch = _single_batch(data).to(device)
        ids, mask = collate_texts(batch, tokenizer, int(cfg["bert"]["max_length"]), device)
        with autocast_ctx(bool(cfg.get("amp", True)), device.type):
            result = model(batch, ids, mask, return_attn=True)
        probs = torch.sigmoid(result["tag_logits"][0].float()).cpu().numpy()
        truth = batch.y_tags.view(-1).cpu().numpy()
        observed = np.isfinite(truth) & (truth != -1)
        f1 = _example_f1(truth[observed], (probs[: observed.sum()] >= 0.5).astype(float)) \
            if observed.any() else float("nan")
        scored.append((i, f1, probs, truth, result, batch, ids, mask))

    valid = [s for s in scored if np.isfinite(s[1])]
    if not valid:
        valid = scored
    valid.sort(key=lambda s: s[1], reverse=True)
    chosen = valid[: max(1, n_cases - 1)] + [valid[-1]]      # best few + the worst
    seen, cases = set(), []

    for case_no, (i, f1, probs, truth, result, batch, ids, mask) in enumerate(chosen[:n_cases]):
        if i in seen:
            continue
        seen.add(i)
        data = dataset[i]
        track = str(getattr(data, "track_id", f"idx{i}"))
        is_failure = bool(np.isfinite(f1) and f1 < 0.5)

        attn = result.get("attn")
        attn_path = None
        if attn is not None:
            tokens = tokenizer.convert_ids_to_tokens(ids[0].cpu())
            keep = int(mask[0].sum().item())
            weights = attn[0].detach().cpu().float().numpy()[:keep]
            attn_path = _plot_cross_attention(
                _clean_tokens(tokens[:keep]), weights,
                out / f"case_study_{case_no}_cross_attention.png",
                title=f"{track} -- graph query attending over caption "
                      f"({'FAILURE' if is_failure else 'success'})",
            )

        edge_alpha = getattr(model.gnn, "last_attention", None)
        graph_path = plot_graph_attention(
            data, edge_alpha=edge_alpha,
            out_path=out / f"case_study_{case_no}_graph.png",
            title=f"{track} -- GATv2 attention per segment",
        )

        top = np.argsort(-probs)[:5]
        true_tags = [bundle.tag_vocab[j] for j in np.flatnonzero(truth > 0)
                     if j < len(bundle.tag_vocab)]
        cases.append({
            "case": case_no,
            "track_id": track,
            "is_failure": is_failure,
            "example_f1": None if not np.isfinite(f1) else float(f1),
            "caption": (getattr(data, "text", "") or "")[:300],
            "predicted_tags": [
                {"tag": bundle.tag_vocab[j], "score": float(probs[j])}
                for j in top if j < len(bundle.tag_vocab)
            ],
            "true_tags": true_tags,
            "cross_attention_plot": str(attn_path) if attn_path else None,
            "graph_plot": str(graph_path),
        })

    n_failures = sum(c["is_failure"] for c in cases)
    LOGGER.info("wrote %d case studies (%d failures) to %s", len(cases), n_failures, out)
    if cases and n_failures == 0:
        LOGGER.warning("no case study scored below F1 0.5 -- the worst example is "
                       "included regardless, but the report should say so")
    return cases


def _plot_cross_attention(tokens, weights, out_path, title: str = ""):
    n = min(len(tokens), len(weights))
    fig, ax = plt.subplots(figsize=(max(6, 0.32 * n), 2.8))
    image = ax.imshow(np.asarray(weights[:n]).reshape(1, -1), aspect="auto", cmap="viridis")
    ax.set_yticks([])
    ax.set_xticks(range(n), tokens[:n], rotation=75, fontsize=7, ha="right")
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.85, label="cross-attention")
    fig.tight_layout()
    out = resolve_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


def _single_batch(data):
    from torch_geometric.data import Batch

    return Batch.from_data_list([data])


def _example_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n = min(y_true.size, y_pred.size)
    y_true, y_pred = y_true[:n], y_pred[:n]
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    if tp + fp + fn == 0:
        return float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
