"""All evaluation metrics. Single source of truth -- nothing else computes one.

Two rules run through every function here:

1. **Sentinels are masked, never scored.** ``-1`` in a tag vector means "this
   dataset does not carry this label", and ``nan`` means the same for a
   regression target. Scoring them as negatives silently rewards a model for
   predicting absence, which is the exact failure this project is designed to
   avoid.
2. **Accuracy is never reported for multi-label tagging.** With a 50-tag
   vocabulary where the median track carries ~4 tags, all-zeros scores ~92%
   element accuracy. Macro-F1, micro-F1 and AUC-PR are the honest numbers.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "tune_thresholds",
    "macro_f1",
    "micro_f1",
    "per_tag_prf",
    "mean_auc_pr",
    "macro_roc_auc",
    "regression_metrics",
    "retrieval_metrics",
    "random_retrieval_reference",
    "graph_coherence_score",
    "knn_probe",
    "multiclass_metrics",
    "confusion_matrix",
    "bootstrap_thresholds",
    "silhouette",
    "aggregate_seeds",
    "TAG_SENTINEL",
]

TAG_SENTINEL = -1.0
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_2d(a) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _valid_mask(y_true: np.ndarray) -> np.ndarray:
    """True where a label is actually observed (not ``-1`` and not ``nan``)."""
    return np.isfinite(y_true) & (y_true != TAG_SENTINEL)


def _prf_from_counts(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall <= 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def _binarise(y_score: np.ndarray, thresholds) -> np.ndarray:
    if thresholds is None:
        thr = np.full(y_score.shape[1], 0.5, dtype=np.float64)
    else:
        thr = np.asarray(thresholds, dtype=np.float64).reshape(-1)
        if thr.size == 1:
            thr = np.full(y_score.shape[1], float(thr[0]))
        if thr.size != y_score.shape[1]:
            raise ValueError(
                f"thresholds has {thr.size} entries but y_score has "
                f"{y_score.shape[1]} tags"
            )
    return (y_score >= thr[None, :]).astype(np.float64)


# --------------------------------------------------------------------------- #
# multi-label classification
# --------------------------------------------------------------------------- #
def tune_thresholds(
    y_true,
    y_score,
    n_grid: int = 101,
    default: float = 0.5,
) -> np.ndarray:
    """Per-tag decision thresholds that maximise F1.

    **Call this on the validation split only.** The returned vector is frozen
    and handed to the test evaluation; tuning on test inflates every number in
    every downstream number and is the easiest way to report a result that
    cannot be reproduced.

    Ties are broken toward the *larger* threshold (more conservative), and a tag
    with no positive validation example keeps ``default``.
    """
    y_true = _as_2d(y_true)
    y_score = _as_2d(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")

    n_tags = y_true.shape[1]
    thresholds = np.full(n_tags, float(default), dtype=np.float64)
    grid = np.linspace(0.0, 1.0, int(n_grid))

    for j in range(n_tags):
        mask = _valid_mask(y_true[:, j])
        if not mask.any():
            continue
        yt = y_true[mask, j]
        ys = y_score[mask, j]
        if yt.sum() == 0:
            continue
        best_f1, best_thr = -1.0, float(default)
        for thr in grid:
            pred = (ys >= thr).astype(np.float64)
            tp = float(np.sum((pred == 1) & (yt == 1)))
            fp = float(np.sum((pred == 1) & (yt == 0)))
            fn = float(np.sum((pred == 0) & (yt == 1)))
            _, _, f1 = _prf_from_counts(tp, fp, fn)
            if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and thr > best_thr):
                best_f1, best_thr = f1, float(thr)
        thresholds[j] = best_thr
    return thresholds


def per_tag_prf(y_true, y_score, thresholds=None) -> pd.DataFrame:
    """Precision / recall / F1 / support / AUC-PR for every tag."""
    y_true = _as_2d(y_true)
    y_score = _as_2d(y_score)
    y_pred = _binarise(y_score, thresholds)
    thr = (
        np.full(y_true.shape[1], 0.5)
        if thresholds is None
        else np.broadcast_to(
            np.asarray(thresholds, dtype=np.float64).reshape(-1), (y_true.shape[1],)
        )
    )

    rows = []
    for j in range(y_true.shape[1]):
        mask = _valid_mask(y_true[:, j])
        yt, yp, ys = y_true[mask, j], y_pred[mask, j], y_score[mask, j]
        tp = float(np.sum((yp == 1) & (yt == 1)))
        fp = float(np.sum((yp == 1) & (yt == 0)))
        fn = float(np.sum((yp == 0) & (yt == 1)))
        precision, recall, f1 = _prf_from_counts(tp, fp, fn)
        rows.append(
            {
                "tag_index": j,
                "threshold": float(thr[j]),
                "support": int(yt.sum()),
                "n_evaluated": int(mask.sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc_pr": _average_precision(yt, ys),
            }
        )
    return pd.DataFrame(rows)


def macro_f1(y_true, y_score, thresholds=None) -> float:
    """Unweighted mean F1 over tags that have at least one observed positive.

    Tags with zero support are excluded rather than scored 0 -- otherwise the
    metric mostly measures how many rare tags happened to land in the split.
    """
    df = per_tag_prf(y_true, y_score, thresholds)
    scored = df[df["support"] > 0]
    if scored.empty:
        return 0.0
    return float(scored["f1"].mean())


def micro_f1(y_true, y_score, thresholds=None) -> float:
    """F1 over all observed (row, tag) cells pooled together."""
    y_true = _as_2d(y_true)
    y_score = _as_2d(y_score)
    y_pred = _binarise(y_score, thresholds)
    mask = _valid_mask(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    tp = float(np.sum((yp == 1) & (yt == 1)))
    fp = float(np.sum((yp == 1) & (yt == 0)))
    fn = float(np.sum((yp == 0) & (yt == 1)))
    return _prf_from_counts(tp, fp, fn)[2]


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-PR for one tag, via sklearn when available (identical convention)."""
    if y_true.size == 0 or y_true.sum() == 0:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_true, y_score))
    except Exception:  # pragma: no cover - fallback keeps metrics importable
        order = np.argsort(-y_score, kind="mergesort")
        yt = y_true[order]
        tp = np.cumsum(yt)
        precision = tp / np.arange(1, yt.size + 1)
        return float(np.sum(precision * yt) / max(yt.sum(), 1))


def mean_auc_pr(y_true, y_score) -> float:
    """Macro-averaged average-precision over tags with observed positives."""
    y_true = _as_2d(y_true)
    y_score = _as_2d(y_score)
    scores = []
    for j in range(y_true.shape[1]):
        mask = _valid_mask(y_true[:, j])
        if not mask.any():
            continue
        value = _average_precision(y_true[mask, j], y_score[mask, j])
        if np.isfinite(value):
            scores.append(value)
    return float(np.mean(scores)) if scores else 0.0


def macro_roc_auc(y_true, y_score) -> float:
    """Macro ROC-AUC over tags that have both classes present."""
    from sklearn.metrics import roc_auc_score

    y_true = _as_2d(y_true)
    y_score = _as_2d(y_score)
    scores = []
    for j in range(y_true.shape[1]):
        mask = _valid_mask(y_true[:, j])
        yt = y_true[mask, j]
        if yt.size == 0 or yt.sum() in (0, yt.size):
            continue
        scores.append(float(roc_auc_score(yt, y_score[mask, j])))
    return float(np.mean(scores)) if scores else 0.0


# --------------------------------------------------------------------------- #
# regression (DEAM valence / arousal)
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred, prefix: str = "") -> dict:
    """MAE / RMSE / R2 with ``nan`` targets dropped.

    R2 uses the mean of the *observed* targets, so it stays comparable across
    batches of different sizes.
    """
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    n = int(mask.sum())
    if n == 0:
        return {f"{prefix}mae": float("nan"), f"{prefix}rmse": float("nan"),
                f"{prefix}r2": float("nan"), f"{prefix}n": 0}
    yt, yp = yt[mask], yp[mask]
    err = yt - yp
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > _EPS else float("nan")
    return {
        f"{prefix}mae": float(np.mean(np.abs(err))),
        f"{prefix}rmse": float(np.sqrt(np.mean(err**2))),
        f"{prefix}r2": float(r2),
        f"{prefix}n": n,
    }


# --------------------------------------------------------------------------- #
# retrieval (Task 4)
# --------------------------------------------------------------------------- #
def random_retrieval_reference(gallery_size: int, ks=(1, 5, 10)) -> dict:
    """What a retrieval model that has learned nothing would score (B2.2).

    Without this row, R@10 = 0.05 reads as a failure. Against a gallery of
    2,503 it is 125x chance, which is a different sentence entirely. Every
    retrieval table needs the reference alongside the gallery size, because the
    two together are what make R@K interpretable at all.

    For a uniformly random ranking of ``N`` candidates containing exactly one
    correct item: ``R@K = K / N``; the true rank is uniform on ``1..N`` so the
    median is ``(N + 1) / 2``; and ``MRR`` is the harmonic number ``H_N / N``,
    which for large ``N`` is close to ``(ln N + 0.5772) / N``.
    """
    n = int(gallery_size)
    if n <= 0:
        return {"gallery_size": 0}
    harmonic = float(np.sum(1.0 / np.arange(1, n + 1)))
    out = {"gallery_size": n, "reference": "uniform random ranking"}
    for k in ks:
        out[f"random_R@{k}"] = float(min(k, n) / n)
    out["random_medR"] = float((n + 1) / 2)
    out["random_meanR"] = float((n + 1) / 2)
    out["random_MRR"] = float(harmonic / n)
    return out


def retrieval_metrics(sim: np.ndarray, ks: Sequence[int] = (1, 5, 10)) -> dict:
    """R@K, median rank and MRR in both directions for a square-ish sim matrix.

    ``sim[i, j]`` is the score between graph *i* and caption *j*; the ground
    truth pairing is the diagonal. Rows index graphs, columns index captions,
    so the caption gallery size is ``sim.shape[1]`` -- reported explicitly
    because R@10 is meaningless without knowing what it was drawn from.
    """
    sim = np.asarray(sim, dtype=np.float64)
    if sim.ndim != 2:
        raise ValueError("similarity matrix must be 2-D")
    n = min(sim.shape)
    out: dict = {
        "gallery_size": int(sim.shape[1]),
        "n_queries": int(sim.shape[0]),
        "n_pairs": int(n),
    }
    if n == 0:
        return out

    def _ranks(matrix: np.ndarray) -> np.ndarray:
        # rank of the true match = how many candidates strictly outscore it,
        # plus one. Strict `>` means ties do not silently inflate R@1.
        idx = np.arange(min(matrix.shape))
        true_scores = matrix[idx, idx][:, None]
        return (matrix[idx] > true_scores).sum(axis=1) + 1

    for name, matrix in (("g2t", sim), ("t2g", sim.T)):
        ranks = _ranks(matrix)
        for k in ks:
            out[f"{name}_R@{k}"] = float(np.mean(ranks <= k))
        out[f"{name}_medR"] = float(np.median(ranks))
        out[f"{name}_meanR"] = float(np.mean(ranks))
        out[f"{name}_MRR"] = float(np.mean(1.0 / ranks))

    for k in ks:
        out[f"mean_R@{k}"] = float((out[f"g2t_R@{k}"] + out[f"t2g_R@{k}"]) / 2)
    out["mean_MRR"] = float((out["g2t_MRR"] + out["t2g_MRR"]) / 2)

    # the chance reference travels with the numbers rather than being
    # looked up later, so no table can report R@K without it.
    reference = random_retrieval_reference(int(sim.shape[1]), ks)
    out.update({k: v for k, v in reference.items() if k.startswith("random_")})
    for k in ks:
        chance = reference.get(f"random_R@{k}", 0.0)
        if chance > 0:
            out[f"mean_R@{k}_vs_chance"] = float(out[f"mean_R@{k}"] / chance)
    return out


# --------------------------------------------------------------------------- #
# representation analysis
# --------------------------------------------------------------------------- #
def graph_coherence_score(h, edge_index, tau: float = 0.5) -> float:
    """S_graph: fraction of edges whose endpoints are more similar than ``tau``.

    From the assignment: S_graph = (1/|E|) * sum_{(i,j) in E} 1[cos(h_i, h_j) > tau].
    Self-loops are excluded -- they are trivially coherent and would just dilute
    the statistic toward 1 as graphs get sparser.
    """
    h = np.asarray(_to_numpy(h), dtype=np.float64)
    edge_index = np.asarray(_to_numpy(edge_index))
    if edge_index.size == 0:
        return float("nan")
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    src, dst = edge_index[0].astype(int), edge_index[1].astype(int)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    if src.size == 0:
        return float("nan")
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    norms = np.maximum(norms, _EPS)
    hn = h / norms
    cos = np.sum(hn[src] * hn[dst], axis=1)
    return float(np.mean(cos > tau))


def knn_probe(emb, labels, k: int = 10, seed: int = 42) -> float:
    """Leave-one-out k-NN label agreement in embedding space.

    The quantitative companion to a t-SNE picture: a cluster plot that "looks
    separated" should also be linearly probe-able. Reported as a single-label
    (genre / mood-quadrant) score, so plain accuracy is legitimate here -- this
    is *not* the multi-label tagging task.
    """
    emb = np.asarray(_to_numpy(emb), dtype=np.float64)
    labels = np.asarray(_to_numpy(labels)).reshape(-1)
    mask = labels != -1
    emb, labels = emb[mask], labels[mask]
    n = emb.shape[0]
    if n < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    k = int(min(k, n - 1))

    norms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), _EPS)
    sim = (emb / norms) @ (emb / norms).T
    np.fill_diagonal(sim, -np.inf)
    neighbours = np.argsort(-sim, axis=1, kind="mergesort")[:, :k]

    correct = 0
    for i in range(n):
        votes, counts = np.unique(labels[neighbours[i]], return_counts=True)
        if votes[np.argmax(counts)] == labels[i]:
            correct += 1
    return float(correct / n)


def confusion_matrix(y_true, y_pred, n_classes: int | None = None) -> np.ndarray:
    """Counts with true classes down the rows, predictions across the columns."""
    y_true = np.asarray(_to_numpy(y_true)).reshape(-1).astype(int)
    y_pred = np.asarray(_to_numpy(y_pred)).reshape(-1).astype(int)
    keep = y_true >= 0
    y_true, y_pred = y_true[keep], y_pred[keep]
    if n_classes is None:
        n_classes = int(max(y_true.max(initial=-1), y_pred.max(initial=-1)) + 1)
    out = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p_ in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p_ < n_classes:
            out[t, p_] += 1
    return out


def multiclass_metrics(y_true, logits, prefix: str = "", n_classes: int | None = None) -> dict:
    """Accuracy, macro-F1 and the confusion matrix for a **single-label** task.

    Accuracy is reported here and nowhere else in this file. The prohibition in
    the module docstring is about multi-label tagging, where all-zeros scores
    ~92% element accuracy and the number is meaningless. FMA-small genre is one
    label per clip over eight classes balanced at ~1,000 clips each, so a chance
    classifier sits at 12.5% and accuracy means exactly what it appears to.
    Macro-F1 travels with it anyway, because the split is only *nearly* balanced
    (997-1,000 per class) and it keeps this row comparable with the tagging
    tables.

    ``y_true`` carries ``-1`` for clips from corpora with no genre label; those
    rows are dropped rather than scored, exactly as the tag sentinel is.
    """
    y_true = np.asarray(_to_numpy(y_true)).reshape(-1).astype(int)
    logits = np.asarray(_to_numpy(logits), dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    keep = y_true >= 0
    if keep.sum() == 0 or logits.shape[0] != y_true.shape[0]:
        return {f"{prefix}accuracy": float("nan"), f"{prefix}macro_f1": float("nan"),
                f"{prefix}n_rows": 0}
    y_true, logits = y_true[keep], logits[keep]
    y_pred = logits.argmax(axis=1)
    n_classes = int(n_classes or logits.shape[1])

    cm = confusion_matrix(y_true, y_pred, n_classes)
    f1s = []
    for c in range(n_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        if cm[c, :].sum() == 0:
            continue                       # class absent from this split
        f1s.append(_prf_from_counts(tp, fp, fn)[2])
    return {
        f"{prefix}accuracy": float((y_pred == y_true).mean()),
        f"{prefix}macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        f"{prefix}n_rows": int(y_true.shape[0]),
        f"{prefix}n_classes": n_classes,
        f"{prefix}per_class_f1": [round(f, 4) for f in f1s],
        f"{prefix}confusion": cm.tolist(),
    }


def bootstrap_thresholds(val_y_true, val_y_score, test_y_true, test_y_score,
                         n_boot: int = 100, seed: int = 42) -> dict:
    """How much do val-tuned thresholds -- and the test score -- move? (A7.4)

    Threshold tuning is a fit like any other, and MTAT's validation split is 977
    clips drawn from 14 artists. If resampling those 977 clips moves the tuned
    operating point a lot, then the tuned test number carries a variance term
    that a single point estimate hides entirely.

    Each of ``n_boot`` replicates resamples the validation rows **with
    replacement**, re-tunes all thresholds on that replicate, and applies them
    unchanged to the *fixed* test split. What comes back is the per-tag
    threshold spread and the resulting test macro-F1 distribution, alongside the
    fixed-0.5 baseline for comparison.
    """
    val_y_true, val_y_score = _as_2d(val_y_true), _as_2d(val_y_score)
    test_y_true, test_y_score = _as_2d(test_y_true), _as_2d(test_y_score)
    n_val, n_tags = val_y_true.shape
    rng = np.random.default_rng(int(seed))

    point = tune_thresholds(val_y_true, val_y_score)
    draws = np.empty((int(n_boot), n_tags), dtype=np.float64)
    macros = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n_val, size=n_val)
        draws[b] = tune_thresholds(val_y_true[idx], val_y_score[idx])
        macros[b] = macro_f1(test_y_true, test_y_score, draws[b])

    finite = macros[np.isfinite(macros)]
    lo, hi = (np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan, np.nan))
    return {
        "n_boot": int(n_boot),
        "n_val_rows": int(n_val),
        "n_tags": int(n_tags),
        "threshold_std_per_tag": draws.std(axis=0).tolist(),
        "threshold_mean_per_tag": draws.mean(axis=0).tolist(),
        "threshold_std_mean": float(draws.std(axis=0).mean()),
        "threshold_std_max": float(draws.std(axis=0).max()),
        "point_thresholds": np.asarray(point, dtype=float).tolist(),
        "test_macro_f1_point": float(macro_f1(test_y_true, test_y_score, point)),
        "test_macro_f1_fixed_half": float(macro_f1(test_y_true, test_y_score, 0.5)),
        "test_macro_f1_mean": float(finite.mean()) if finite.size else float("nan"),
        "test_macro_f1_std": float(finite.std()) if finite.size else float("nan"),
        "test_macro_f1_min": float(finite.min()) if finite.size else float("nan"),
        "test_macro_f1_max": float(finite.max()) if finite.size else float("nan"),
        "test_macro_f1_spread": float(finite.max() - finite.min()) if finite.size else float("nan"),
        "test_macro_f1_ci95": [float(lo), float(hi)],
    }


def silhouette(emb, labels) -> float:
    """Mean silhouette over cosine distances; ``nan`` if fewer than 2 clusters."""
    from sklearn.metrics import silhouette_score

    emb = np.asarray(_to_numpy(emb), dtype=np.float64)
    labels = np.asarray(_to_numpy(labels)).reshape(-1)
    mask = labels != -1
    emb, labels = emb[mask], labels[mask]
    if emb.shape[0] < 3 or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(silhouette_score(emb, labels, metric="cosine"))
    except Exception:  # pragma: no cover - degenerate embeddings
        return float("nan")


# --------------------------------------------------------------------------- #
# multi-seed aggregation
# --------------------------------------------------------------------------- #
def aggregate_seeds(runs: Iterable[dict]) -> dict:
    """mean / std / n for every numeric key shared across seed runs.

    Non-numeric values are kept only when identical across runs, so a summary
    still carries the model name but never a misleading "mean" of strings.
    """
    runs = [r for r in runs if isinstance(r, dict)]
    if not runs:
        return {}
    out: dict = {"n_seeds": len(runs)}
    keys: list[str] = []
    for run in runs:
        for key in run:
            if key not in keys:
                keys.append(key)

    for key in keys:
        values = [r[key] for r in runs if key in r]
        numeric = [
            float(v)
            for v in values
            if isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, bool)
            and np.isfinite(float(v))
        ]
        if numeric and len(numeric) == len(values):
            out[f"{key}_mean"] = float(np.mean(numeric))
            out[f"{key}_std"] = float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0
            out[f"{key}_n"] = len(numeric)
        elif len({str(v) for v in values}) == 1:
            out[key] = values[0]
    return out


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return x
