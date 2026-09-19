"""Metrics are checked against hand-computed values, not against themselves.

Every number in this file was worked out on paper first; if an implementation
changes and these fail, the implementation is wrong, not the test.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import metrics as M


# --------------------------------------------------------------------------- #
# F1
# --------------------------------------------------------------------------- #
def test_micro_f1_hand_computed():
    # 2 tags x 4 rows. Threshold 0.5.
    #   tag0 preds [1,1,0,0] truth [1,0,1,0] -> tp1 fp1 fn1
    #   tag1 preds [1,0,1,0] truth [1,1,0,0] -> tp1 fp1 fn1
    # pooled: tp=2 fp=2 fn=2 -> P=R=0.5 -> F1=0.5
    y_true = np.array([[1, 1], [0, 1], [1, 0], [0, 0]], dtype=float)
    y_score = np.array([[0.9, 0.9], [0.8, 0.1], [0.2, 0.7], [0.1, 0.2]])
    assert M.micro_f1(y_true, y_score) == pytest.approx(0.5)


def test_macro_f1_hand_computed():
    # tag0: tp1 fp1 fn1 -> F1 = 2*.5*.5/1 = 0.5
    # tag1: tp1 fp1 fn1 -> F1 = 0.5
    # macro = 0.5
    y_true = np.array([[1, 1], [0, 1], [1, 0], [0, 0]], dtype=float)
    y_score = np.array([[0.9, 0.9], [0.8, 0.1], [0.2, 0.7], [0.1, 0.2]])
    assert M.macro_f1(y_true, y_score) == pytest.approx(0.5)


def test_macro_f1_perfect_and_zero():
    y_true = np.array([[1, 0], [0, 1]], dtype=float)
    assert M.macro_f1(y_true, np.array([[0.9, 0.1], [0.1, 0.9]])) == pytest.approx(1.0)
    assert M.macro_f1(y_true, np.array([[0.1, 0.9], [0.9, 0.1]])) == pytest.approx(0.0)


def test_macro_f1_excludes_zero_support_tags():
    # tag1 has no positives anywhere: it must be skipped, not scored as 0,
    # otherwise macro-F1 measures split composition rather than the model.
    y_true = np.array([[1, 0], [1, 0], [0, 0]], dtype=float)
    y_score = np.array([[0.9, 0.2], [0.8, 0.1], [0.1, 0.3]])
    assert M.macro_f1(y_true, y_score) == pytest.approx(1.0)


def test_macro_micro_diverge_on_imbalance():
    """A frequent tag done well and a rare tag done badly must separate."""
    rng = np.random.default_rng(0)
    y_true = np.zeros((100, 2))
    y_true[:80, 0] = 1        # frequent tag
    y_true[:2, 1] = 1         # rare tag
    y_score = np.zeros((100, 2))
    y_score[:80, 0] = 0.9     # predicted perfectly
    y_score[:, 1] = 0.1       # rare tag never predicted
    assert M.micro_f1(y_true, y_score) > M.macro_f1(y_true, y_score)


# --------------------------------------------------------------------------- #
# sentinel masking
# --------------------------------------------------------------------------- #
def test_sentinel_minus_one_rows_are_ignored():
    """A row of ``-1`` must not change any score."""
    y_true = np.array([[1, 1], [0, 1], [1, 0], [0, 0]], dtype=float)
    y_score = np.array([[0.9, 0.9], [0.8, 0.1], [0.2, 0.7], [0.1, 0.2]])
    baseline = M.macro_f1(y_true, y_score)

    padded_true = np.vstack([y_true, np.full((3, 2), -1.0)])
    padded_score = np.vstack([y_score, np.full((3, 2), 0.99)])
    assert M.macro_f1(padded_true, padded_score) == pytest.approx(baseline)
    assert M.micro_f1(padded_true, padded_score) == pytest.approx(
        M.micro_f1(y_true, y_score)
    )


def test_sentinel_masking_is_per_cell():
    y_true = np.array([[1, -1], [0, 1], [1, -1]], dtype=float)
    y_score = np.array([[0.9, 0.9], [0.1, 0.9], [0.8, 0.9]])
    frame = M.per_tag_prf(y_true, y_score)
    assert int(frame.loc[frame.tag_index == 0, "n_evaluated"].iloc[0]) == 3
    assert int(frame.loc[frame.tag_index == 1, "n_evaluated"].iloc[0]) == 1


def test_all_sentinel_column_scores_nothing():
    y_true = np.full((4, 2), -1.0)
    y_score = np.random.default_rng(0).random((4, 2))
    assert M.macro_f1(y_true, y_score) == 0.0
    assert M.mean_auc_pr(y_true, y_score) == 0.0


# --------------------------------------------------------------------------- #
# thresholds
# --------------------------------------------------------------------------- #
def test_tune_thresholds_recovers_a_separating_cut():
    y_true = np.array([[0], [0], [1], [1]], dtype=float)
    y_score = np.array([[0.10], [0.20], [0.80], [0.90]])
    thresholds = M.tune_thresholds(y_true, y_score)
    assert 0.2 < thresholds[0] <= 0.8
    assert M.macro_f1(y_true, y_score, thresholds) == pytest.approx(1.0)


def test_tune_thresholds_defaults_when_a_tag_has_no_positives():
    y_true = np.array([[0], [0], [0]], dtype=float)
    y_score = np.array([[0.4], [0.5], [0.6]])
    assert M.tune_thresholds(y_true, y_score, default=0.5)[0] == pytest.approx(0.5)


def test_tune_thresholds_is_per_tag():
    # tag0 separates at ~0.5, tag1 at ~0.15 -- one global cut cannot do both
    y_true = np.array([[0, 0], [0, 1], [1, 1], [1, 1]], dtype=float)
    y_score = np.array([[0.10, 0.05], [0.20, 0.20], [0.70, 0.30], [0.90, 0.40]])
    thresholds = M.tune_thresholds(y_true, y_score)
    assert thresholds[0] > thresholds[1]
    assert M.macro_f1(y_true, y_score, thresholds) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# AUC-PR / ROC-AUC
# --------------------------------------------------------------------------- #
def test_mean_auc_pr_perfect_ranking():
    y_true = np.array([[1], [1], [0], [0]], dtype=float)
    y_score = np.array([[0.9], [0.8], [0.2], [0.1]])
    assert M.mean_auc_pr(y_true, y_score) == pytest.approx(1.0)


def test_mean_auc_pr_hand_computed_partial():
    # ranking 1, 0, 1, 0: AP = (1/1 * 1 + 2/3 * 1) / 2 = 0.8333...
    y_true = np.array([[1], [0], [1], [0]], dtype=float)
    y_score = np.array([[0.9], [0.8], [0.7], [0.6]])
    assert M.mean_auc_pr(y_true, y_score) == pytest.approx(5.0 / 6.0, abs=1e-6)


def test_macro_roc_auc_perfect():
    y_true = np.array([[1], [1], [0], [0]], dtype=float)
    y_score = np.array([[0.9], [0.8], [0.2], [0.1]])
    assert M.macro_roc_auc(y_true, y_score) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# regression
# --------------------------------------------------------------------------- #
def test_regression_metrics_hand_computed():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.5, 2.5, 4.5])
    out = M.regression_metrics(y_true, y_pred)
    assert out["mae"] == pytest.approx(0.5)
    assert out["rmse"] == pytest.approx(0.5)
    # ss_res = 4 * 0.25 = 1.0 ; ss_tot = 1.25+0.25+0.25+2.25... -> 5.0
    assert out["r2"] == pytest.approx(1.0 - 1.0 / 5.0)
    assert out["n"] == 4


def test_regression_metrics_masks_nan_targets():
    y_true = np.array([1.0, np.nan, 3.0, np.nan])
    y_pred = np.array([1.5, 99.0, 2.5, -99.0])
    out = M.regression_metrics(y_true, y_pred)
    assert out["n"] == 2
    assert out["mae"] == pytest.approx(0.5)
    assert np.isfinite(out["rmse"])


def test_regression_metrics_all_nan_is_nan_not_zero():
    out = M.regression_metrics(np.array([np.nan, np.nan]), np.array([1.0, 2.0]))
    assert out["n"] == 0
    assert np.isnan(out["mae"])


# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #
def test_retrieval_metrics_identity_matrix_is_perfect():
    sim = np.eye(5)
    out = M.retrieval_metrics(sim)
    for key in ("g2t_R@1", "t2g_R@1", "mean_R@1", "g2t_MRR"):
        assert out[key] == pytest.approx(1.0)
    assert out["g2t_medR"] == pytest.approx(1.0)
    assert out["gallery_size"] == 5


def test_retrieval_metrics_hand_computed_ranks():
    # row 0: true score 0.1, two candidates beat it -> rank 3
    # row 1: true score 0.9, nothing beats it       -> rank 1
    # row 2: true score 0.5, one candidate beats it -> rank 2
    sim = np.array([
        [0.1, 0.5, 0.9],
        [0.2, 0.9, 0.3],
        [0.7, 0.1, 0.5],
    ])
    out = M.retrieval_metrics(sim)
    assert out["g2t_R@1"] == pytest.approx(1 / 3)
    assert out["g2t_R@5"] == pytest.approx(1.0)
    assert out["g2t_medR"] == pytest.approx(2.0)
    assert out["g2t_MRR"] == pytest.approx((1 / 3 + 1 / 1 + 1 / 2) / 3)


def test_retrieval_metrics_reports_gallery_size():
    sim = np.random.default_rng(0).random((10, 40))
    assert M.retrieval_metrics(sim)["gallery_size"] == 40
    assert M.retrieval_metrics(sim)["n_queries"] == 10


# --------------------------------------------------------------------------- #
# S_graph
# --------------------------------------------------------------------------- #
def test_graph_coherence_all_identical_nodes():
    h = np.ones((4, 3))
    edge_index = np.array([[0, 1, 2], [1, 2, 3]])
    assert M.graph_coherence_score(h, edge_index, tau=0.5) == pytest.approx(1.0)


def test_graph_coherence_orthogonal_nodes():
    h = np.eye(4)
    edge_index = np.array([[0, 1, 2], [1, 2, 3]])
    assert M.graph_coherence_score(h, edge_index, tau=0.5) == pytest.approx(0.0)


def test_graph_coherence_hand_computed_fraction():
    # nodes 0,1 identical (cos 1); node 2 orthogonal to both (cos 0).
    # edges (0,1) coherent, (1,2) and (0,2) not -> 1/3
    h = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    edge_index = np.array([[0, 1, 0], [1, 2, 2]])
    assert M.graph_coherence_score(h, edge_index, tau=0.5) == pytest.approx(1 / 3)


def test_graph_coherence_ignores_self_loops():
    # without excluding self-loops, the trivially coherent (i,i) edges would
    # drag the statistic toward 1 on any sparse graph
    h = np.array([[1.0, 0.0], [0.0, 1.0]])
    with_loops = np.array([[0, 1, 0, 1], [1, 0, 0, 1]])
    assert M.graph_coherence_score(h, with_loops, tau=0.5) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# representation probes and aggregation
# --------------------------------------------------------------------------- #
def test_knn_probe_separable_clusters():
    a = np.random.default_rng(0).normal(0, 0.01, size=(20, 4)) + np.array([5, 0, 0, 0])
    b = np.random.default_rng(1).normal(0, 0.01, size=(20, 4)) + np.array([0, 5, 0, 0])
    emb = np.vstack([a, b])
    labels = np.array([0] * 20 + [1] * 20)
    assert M.knn_probe(emb, labels, k=5) > 0.95


def test_knn_probe_ignores_sentinel_labels():
    emb = np.random.default_rng(0).normal(size=(10, 3))
    labels = np.array([-1] * 10)
    assert np.isnan(M.knn_probe(emb, labels))


def test_silhouette_separable_is_positive():
    a = np.random.default_rng(0).normal(0, 0.05, size=(15, 3)) + np.array([3, 0, 0])
    b = np.random.default_rng(1).normal(0, 0.05, size=(15, 3)) + np.array([0, 3, 0])
    labels = np.array([0] * 15 + [1] * 15)
    assert M.silhouette(np.vstack([a, b]), labels) > 0.5


def test_aggregate_seeds_mean_and_std():
    runs = [{"macro_f1": 0.4}, {"macro_f1": 0.5}, {"macro_f1": 0.6}]
    out = M.aggregate_seeds(runs)
    assert out["n_seeds"] == 3
    assert out["macro_f1_mean"] == pytest.approx(0.5)
    assert out["macro_f1_std"] == pytest.approx(0.1)


def test_aggregate_seeds_keeps_constant_strings_only():
    runs = [{"model": "gnn", "seed_name": "a"}, {"model": "gnn", "seed_name": "b"}]
    out = M.aggregate_seeds(runs)
    assert out["model"] == "gnn"
    assert "seed_name" not in out


def test_no_accuracy_function_is_exported():
    """Guardrail: accuracy must not be reachable from the metrics module."""
    assert not [name for name in dir(M) if "accuracy" in name.lower()]


# --------------------------------------------------------------------------- #
# single-label multiclass metrics
# --------------------------------------------------------------------------- #
def test_multiclass_metrics_perfect_prediction():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    logits = np.eye(4)[y] * 10.0
    out = M.multiclass_metrics(y, logits)
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["macro_f1"] == pytest.approx(1.0)
    assert out["n_rows"] == 8


def test_multiclass_metrics_drops_the_minus_one_sentinel():
    """Clips from corpora with no genre column must not be scored as class 0."""
    y = np.array([-1, -1, 1, 1])
    logits = np.tile([5.0, 0.0], (4, 1))          # everything predicted class 0
    out = M.multiclass_metrics(y, logits)
    assert out["n_rows"] == 2, "sentinel rows leaked into the score"
    assert out["accuracy"] == pytest.approx(0.0)


def test_multiclass_confusion_matrix_rows_are_truth():
    y = np.array([0, 0, 1])
    logits = np.array([[9.0, 0.0], [0.0, 9.0], [0.0, 9.0]])
    cm = np.asarray(M.multiclass_metrics(y, logits)["confusion"])
    assert cm.tolist() == [[1, 1], [0, 1]]


def test_multiclass_chance_level_is_near_one_over_k():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 8, 4000)
    out = M.multiclass_metrics(y, rng.normal(size=(4000, 8)))
    assert 0.09 < out["accuracy"] < 0.16, "chance should sit near 1/8"


# --------------------------------------------------------------------------- #
# bootstrapped threshold stability
# --------------------------------------------------------------------------- #
def _separable(n_rows, n_tags, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n_rows, n_tags)) < 0.3).astype(float)
    score = np.clip(y * 0.6 + rng.normal(0, noise * 0.15, y.shape) + 0.2, 0.01, 0.99)
    return y, score


def test_bootstrap_thresholds_reports_one_std_per_tag():
    yv, sv = _separable(200, 6, seed=1)
    yt, st = _separable(200, 6, seed=2)
    out = M.bootstrap_thresholds(yv, sv, yt, st, n_boot=15, seed=0)
    assert len(out["threshold_std_per_tag"]) == 6
    assert len(out["threshold_mean_per_tag"]) == 6
    assert out["n_boot"] == 15 and out["n_val_rows"] == 200


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    yv, sv = _separable(120, 4, seed=3)
    yt, st = _separable(120, 4, seed=4)
    a = M.bootstrap_thresholds(yv, sv, yt, st, n_boot=10, seed=7)
    b = M.bootstrap_thresholds(yv, sv, yt, st, n_boot=10, seed=7)
    assert a["test_macro_f1_mean"] == pytest.approx(b["test_macro_f1_mean"])
    assert a["threshold_std_per_tag"] == b["threshold_std_per_tag"]


def test_bootstrap_spread_is_larger_on_a_smaller_validation_split():
    """The whole point of A7.4: a small val split makes tuning noisier.

    977 clips from 14 artists is what MTAT actually gives us, so the claim that
    the spread depends on validation size has to hold in the code, not just in
    the argument.
    """
    yt, st = _separable(600, 8, seed=11)
    big = M.bootstrap_thresholds(*_separable(600, 8, seed=10), yt, st,
                                 n_boot=25, seed=0)
    small = M.bootstrap_thresholds(*_separable(60, 8, seed=10), yt, st,
                                   n_boot=25, seed=0)
    assert small["test_macro_f1_std"] > big["test_macro_f1_std"]


def test_bootstrap_reports_the_fixed_half_baseline_too():
    yv, sv = _separable(150, 5, seed=5)
    yt, st = _separable(150, 5, seed=6)
    out = M.bootstrap_thresholds(yv, sv, yt, st, n_boot=10, seed=0)
    assert out["test_macro_f1_fixed_half"] == pytest.approx(
        M.macro_f1(yt, st, 0.5))
    assert np.isfinite(out["test_macro_f1_point"])
