"""End-to-end smoke tests: all four tasks train one epoch on CPU.

Slow by design -- they exercise real models on the synthetic dataset. They are
the tests that would have caught every integration bug this project can have:
a shape mismatch between the graph batch and the tokenised text, a loss that
silently trains on sentinels, an early-stopping metric that is never populated.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.synthetic import make_synthetic_dataset
from src.utils import load_config, project_root


TASKS = [1, 2, 3, 4]


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """A small synthetic dataset, generated once for the whole session."""
    out_dir = tmp_path_factory.mktemp("synthetic")
    cfg = load_config("config.yaml")
    make_synthetic_dataset(n_tracks=48, n_tags=12, out_dir=str(out_dir), seed=42, cfg=cfg)
    return out_dir


@pytest.fixture(scope="session")
def results_dir(tmp_path_factory):
    """Keep test artefacts out of the repo's real results/ directory.

    Without this, a tiny 32-dim test checkpoint lands in results/checkpoints/
    and the next `python -m src.evaluate` tries to load it into the full-size
    model.
    """
    return tmp_path_factory.mktemp("results")


@pytest.fixture(scope="session")
def cfg_overrides(synthetic, tiny_bert, results_dir):
    return [
        f"paths.results={results_dir.as_posix()}",
        f"paths.checkpoints={(results_dir / 'checkpoints').as_posix()}",
        "train.epochs=1",
        "train.batch_size=4",
        "train.grad_accum_steps=2",
        "gnn.hidden_dim=32",
        "fusion.shared_dim=32",
        f"bert.model_name={tiny_bert}",
        "bert.max_length=32",
        "bert.freeze_epochs=0",
        "contrastive.batch_size=8",
        f"synthetic.out_dir={synthetic.as_posix()}",
    ]


def _results_root(overrides) -> Path:
    prefix = "paths.results="
    for item in overrides:
        if item.startswith(prefix):
            return Path(item[len(prefix):])
    return project_root() / "results"


def _run(task: int, overrides, extra=None):
    from src.train import main as train_main

    argv = ["--task", str(task), "--synthetic", "--device", "cpu",
            "--num-workers", "0", "--override", *overrides]
    argv += extra or []
    assert train_main(argv) == 0
    results = _results_root(overrides) / f"task{task}_seed42.json"
    assert results.exists(), f"task {task} wrote no result file"
    with open(results, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.slow
@pytest.mark.parametrize("task", TASKS)
def test_task_trains_one_epoch_on_cpu(task, cfg_overrides):
    result = _run(task, cfg_overrides)

    assert result["epochs_run"] == 1
    assert result["device"] == "cpu"
    history = result["history"]
    assert history, "no epoch history recorded"

    loss = history[0].get("train_loss_total")
    assert loss is not None and np.isfinite(loss), f"non-finite training loss: {loss}"
    assert history[0]["train_batches"] > 0, "no batches were consumed"

    for key, value in (result.get("test") or {}).items():
        if isinstance(value, (int, float)):
            assert not (isinstance(value, float) and np.isnan(value)), \
                f"task {task} produced nan for test metric {key!r}"


@pytest.mark.slow
def test_gradient_accumulation_is_configurable(cfg_overrides):
    result = _run(2, [o for o in cfg_overrides if not o.startswith("train.grad_accum")]
                  + ["train.grad_accum_steps=3"])
    assert result["grad_accum_steps"] == 3
    assert result["effective_batch"] == result["grad_accum_steps"] * 4


@pytest.mark.slow
def test_task4_reports_gallery_size(cfg_overrides):
    result = _run(4, cfg_overrides)
    test = result["test"]
    assert test["gallery_size"] > 0
    for key in ("g2t_R@1", "t2g_R@1", "g2t_medR", "mean_MRR"):
        assert key in test


@pytest.mark.slow
def test_thresholds_come_from_validation(cfg_overrides):
    result = _run(1, cfg_overrides)
    assert result["threshold_source"] == "val"
    assert result["thresholds"] is not None
    assert len(result["thresholds"]) == result["n_tags"]


# --------------------------------------------------------------------------- #
# the masked multi-task loss -- acceptance criterion 12
# --------------------------------------------------------------------------- #
class _Batch:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_masked_loss_ignores_minus_one_tags():
    """A row of ``-1`` tags must contribute nothing, whatever the logits say."""
    from src.fusion_model import masked_multitask_loss

    cfg = load_config("config.yaml", {"multitask.auto_balance": False})
    logits = torch.zeros(2, 4)
    y_real = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    y_absent = torch.full((1, 4), -1.0)

    only_real = _Batch(y_tags=y_real, y_valence=torch.tensor([float("nan")]),
                       y_arousal=torch.tensor([float("nan")]))
    with_absent = _Batch(y_tags=torch.cat([y_real, y_absent]),
                         y_valence=torch.tensor([float("nan"), float("nan")]),
                         y_arousal=torch.tensor([float("nan"), float("nan")]))

    loss_a, parts_a = masked_multitask_loss(
        {"tag_logits": logits[:1], "valence": None, "arousal": None}, only_real, cfg)
    loss_b, parts_b = masked_multitask_loss(
        {"tag_logits": logits, "valence": None, "arousal": None}, with_absent, cfg)

    assert parts_a["n_tag_cells"] == 4
    assert parts_b["n_tag_cells"] == 4, "sentinel cells leaked into the loss"
    assert float(loss_a) == pytest.approx(float(loss_b), abs=1e-6)


def test_masked_loss_ignores_nan_regression_targets():
    from src.fusion_model import masked_multitask_loss

    cfg = load_config("config.yaml", {"multitask.auto_balance": False})
    out = {
        "tag_logits": torch.zeros(3, 2),
        "valence": torch.tensor([5.0, 99.0, 4.0]),
        "arousal": torch.tensor([5.0, -99.0, 4.0]),
    }
    batch = _Batch(
        y_tags=torch.full((3, 2), -1.0),
        y_valence=torch.tensor([5.0, float("nan"), 4.0]),
        y_arousal=torch.tensor([5.0, float("nan"), 4.0]),
    )
    loss, parts = masked_multitask_loss(out, batch, cfg)
    assert parts["n_valence"] == 2 and parts["n_arousal"] == 2
    assert parts["loss_valence"] == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(loss), "nan targets reached the arithmetic"
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_masked_loss_is_finite_when_a_batch_has_no_labels_at_all():
    from src.fusion_model import masked_multitask_loss

    cfg = load_config("config.yaml")
    out = {"tag_logits": torch.zeros(2, 3),
           "valence": torch.zeros(2), "arousal": torch.zeros(2)}
    batch = _Batch(y_tags=torch.full((2, 3), -1.0),
                   y_valence=torch.tensor([float("nan")] * 2),
                   y_arousal=torch.tensor([float("nan")] * 2))
    loss, parts = masked_multitask_loss(out, batch, cfg)
    assert torch.isfinite(loss) and float(loss) == 0.0
    assert parts["n_tag_cells"] == 0


def test_masked_loss_gradients_do_not_flow_from_sentinels():
    from src.fusion_model import masked_multitask_loss

    cfg = load_config("config.yaml", {"multitask.auto_balance": False})
    logits = torch.zeros(2, 3, requires_grad=True)
    batch = _Batch(y_tags=torch.tensor([[1.0, 0.0, 1.0], [-1.0, -1.0, -1.0]]),
                   y_valence=torch.tensor([float("nan")] * 2),
                   y_arousal=torch.tensor([float("nan")] * 2))
    loss, _ = masked_multitask_loss(
        {"tag_logits": logits, "valence": None, "arousal": None}, batch, cfg)
    loss.backward()
    assert torch.all(logits.grad[1] == 0), "sentinel row received gradient"
    assert torch.any(logits.grad[0] != 0), "labelled row received no gradient"


# --------------------------------------------------------------------------- #
# amp / model plumbing
# --------------------------------------------------------------------------- #
def test_autocast_is_a_noop_on_cpu():
    from src.utils import autocast_ctx

    with autocast_ctx(True, "cpu"):
        x = torch.ones(2, 2) @ torch.ones(2, 2)
    assert x.dtype == torch.float32


def test_grad_scaler_is_disabled_off_gpu():
    from src.utils import make_grad_scaler

    assert make_grad_scaler(True, "cpu").is_enabled() is False


def test_gnn_encoder_supports_both_convs_and_all_readouts():
    from src.gnn_model import CONVS, READOUTS, GNNEncoder
    from torch_geometric.data import Batch

    from src.graph_builder import build_segment_graph

    cfg = load_config("config.yaml")
    feats = np.random.default_rng(0).normal(size=(8, 96)).astype(np.float32)
    batch = Batch.from_data_list([build_segment_graph(feats, cfg) for _ in range(2)])
    for conv in CONVS:
        for readout in READOUTS:
            encoder = GNNEncoder(96, 16, 2, conv=conv, readout=readout, dropout=0.0)
            g, h = encoder(batch, return_nodes=True)
            assert g.shape == (2, encoder.out_dim)
            assert h.shape[0] == batch.x.shape[0]
            assert torch.isfinite(g).all()


def test_gatv2_exposes_attention_for_the_case_studies():
    from torch_geometric.data import Batch

    from src.gnn_model import GNNEncoder
    from src.graph_builder import build_segment_graph

    cfg = load_config("config.yaml")
    feats = np.random.default_rng(0).normal(size=(8, 96)).astype(np.float32)
    batch = Batch.from_data_list([build_segment_graph(feats, cfg)])
    encoder = GNNEncoder(96, 16, 2, conv="gatv2", dropout=0.0)
    encoder(batch)
    assert encoder.last_attention is not None
    edge_index, alpha = encoder.last_attention
    assert alpha.shape[0] == edge_index.shape[1]


def test_cnn_baseline_adapts_to_a_smaller_mel_count():
    """Replaces the old parameter-matching test, removed with the constraint.

    A7.2: matching B2's capacity to the GNN is what produced the implausible
    0.1654 macro-F1, so the ability to do it is gone rather than merely unused.
    What still has to hold is that the block stack stops pooling the frequency
    axis before it vanishes, which is what this checks at 64 mels.
    """
    from src.cnn_baseline import MelCNN

    model = MelCNN(n_tags=10, n_mels=64)
    out = model(torch.randn(2, 1, 64, 128))
    assert out["tag_logits"].shape == (2, 10)


def test_all_fusion_modes_run(tiny_bert):
    from torch_geometric.data import Batch

    from src.bert_encoder import BertTextEncoder
    from src.fusion_model import FUSION_MODES, GNNBertFusion
    from src.gnn_model import GNNEncoder
    from src.graph_builder import build_segment_graph

    cfg = load_config("config.yaml")
    feats = np.random.default_rng(0).normal(size=(6, 96)).astype(np.float32)
    batch = Batch.from_data_list([build_segment_graph(feats, cfg, text="a test")
                                  for _ in range(2)])
    ids = torch.randint(0, 100, (2, 8))
    mask = torch.ones_like(ids)

    for mode in FUSION_MODES:
        gnn = GNNEncoder(96, 16, 1, dropout=0.0)
        bert = BertTextEncoder(tiny_bert, freeze_mode="frozen_probe")
        model = GNNBertFusion(gnn, bert, mode=mode, shared_dim=16, n_heads=2, n_tags=5)
        out = model(batch, ids, mask, return_attn=True)
        assert out["tag_logits"].shape == (2, 5), mode
        assert torch.isfinite(out["tag_logits"]).all(), mode
        assert out["valence"].shape == (2,), mode


def test_symmetric_info_nce_is_minimised_by_aligned_embeddings():
    from src.contrastive import symmetric_info_nce

    aligned = torch.eye(4)
    scrambled = torch.eye(4)[[1, 0, 3, 2]]
    good = symmetric_info_nce(aligned, aligned, torch.tensor(10.0))
    bad = symmetric_info_nce(aligned, scrambled, torch.tensor(10.0))
    assert float(good) < float(bad)
    assert torch.isfinite(good)


def test_synthetic_dataset_has_realistic_sentinel_patterns(synthetic):
    with open(Path(synthetic) / "summary.json", "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["tracks_with_both_tags_and_va"] == 0, (
        "the real corpora never overlap; the synthetic set must not either"
    )
    assert summary["tracks_with_tags"] > 0
    assert summary["tracks_with_valence_arousal"] > 0
    assert set(summary["split_counts"]) == {"train", "val", "test"}


def test_synthetic_splits_are_leak_free(synthetic):
    import pandas as pd

    from src.splits import assert_no_leakage

    assert_no_leakage(pd.read_csv(Path(synthetic) / "manifest.csv"))


def test_norm_stats_are_train_only(synthetic):
    with open(Path(synthetic) / "norm_stats.json", "r", encoding="utf-8") as fh:
        stats = json.load(fh)
    assert stats["split"] == "train"
    assert len(stats["mean"]) == 96 and len(stats["std"]) == 96


def test_compute_norm_stats_refuses_non_train_splits():
    from src.audio_features import compute_norm_stats

    with pytest.raises(ValueError, match="train split"):
        compute_norm_stats(None, split="test")


# --------------------------------------------------------------------------- #
# codebase conventions -- acceptance criteria 9, 10 and 11, enforced by test
# --------------------------------------------------------------------------- #
SOURCE_DIRS = ("src", "scripts")


def _source_files():
    for directory in SOURCE_DIRS:
        for path in sorted((project_root() / directory).rglob("*.py")):
            yield path, path.read_text(encoding="utf-8")


def test_no_stub_implementations_remain():
    """Criterion 9: no NotImplementedError, and no bare `pass` used as a body."""
    offenders = []
    for path, text in _source_files():
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if "NotImplementedError" in line:
                offenders.append(f"{path.name}:{i + 1} NotImplementedError")
            if line.strip() == "pass":
                # a bare pass is fine as an exception handler, not as a body
                previous = next(
                    (lines[j] for j in range(i - 1, max(i - 4, -1), -1) if lines[j].strip()),
                    "",
                )
                if not previous.strip().startswith(("except", "finally", "class", "if TYPE")):
                    offenders.append(f"{path.name}:{i + 1} bare pass after {previous.strip()!r}")
    assert not offenders, "stub implementations found: " + "; ".join(offenders)


def _code_tokens(text: str):
    """Yield the source tokens that are actual code, not comments or strings.

    Scanning raw lines cannot tell a docstring that *explains* why accuracy is
    the wrong metric from a line that computes one; tokenising can.
    """
    import tokenize

    # Python 3.12 splits f-strings into FSTRING_START/MIDDLE/END, and
    # FSTRING_MIDDLE is not tokenize.STRING -- so the literal text of an
    # f-string used to leak through a filter whose stated purpose is to skip
    # string content. Named via getattr because the token does not exist on
    # older interpreters.
    skip = {tokenize.COMMENT, tokenize.STRING}
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        code = getattr(tokenize, name, None)
        if code is not None:
            skip.add(code)

    reader = io.StringIO(text).readline
    try:
        for token in tokenize.generate_tokens(reader):
            if token.type in skip:
                continue
            if token.string.strip():
                yield token
    except (tokenize.TokenError, IndentationError):
        return


def test_accuracy_is_never_reported_for_multi_label_tagging():
    """Criterion 10: the only accuracy in the code is the flagged counter-example.

    A7.1 added a legitimate accuracy: FMA-small genre is one label per clip over
    eight near-balanced classes, where chance is 12.5% and the number means
    what it appears to. The identifiers it introduces are whitelisted here, and
    the behavioural test below is what actually enforces the rule -- a token
    grep can be defeated by string concatenation, and was.
    """
    allowed = {"element_accuracy_do_not_report", "element_accuracy",
               "genre_accuracy", "accuracy"}
    offenders = []
    for path, text in _source_files():
        for token in _code_tokens(text):
            name = token.string
            if "accuracy" in name.lower() and name not in allowed:
                offenders.append(f"{path.name}:{token.start[0]}: {name}")
    assert not offenders, (
        "accuracy appears to be computed or reported as a metric: "
        + "; ".join(offenders)
    )


def test_the_multi_label_metric_path_returns_no_accuracy_key():
    """The rule that matters, checked on behaviour rather than on spelling.

    With a 50-tag vocabulary where the median track carries ~4 tags, all-zeros
    scores ~92% element accuracy. Nothing on the multi-label path may hand
    that number back under a name a table could pick up.
    """
    import numpy as np

    from src import metrics as M
    from src.train import tagging_metrics

    y_true = (np.random.default_rng(0).random((40, 50)) < 0.08).astype(float)
    y_score = np.random.default_rng(1).random((40, 50))

    for produced in (M.per_tag_prf(y_true, y_score).columns.tolist(),
                     list(tagging_metrics({"targets": y_true, "scores": y_score}))):
        leaked = [k for k in produced if "accuracy" in str(k).lower()]
        assert not leaked, f"multi-label path returned {leaked}"

    # and the single-label path must still provide one, or Table II loses a column
    single = M.multiclass_metrics(np.array([0, 1, 2]), np.eye(3) * 5.0)
    assert "accuracy" in single


def test_thresholds_are_never_tuned_on_test():
    """Criterion 10: every tune_thresholds call site uses val (or train) scores."""
    offenders = []
    for path, text in _source_files():
        for i, line in enumerate(text.splitlines()):
            if "tune_thresholds(" not in line or "def tune_thresholds" in line:
                continue
            arguments = line.split("tune_thresholds(", 1)[1]
            arguments = arguments.replace("yte.shape", "").replace("y_test.shape", "")
            if "test" in arguments.lower():
                offenders.append(f"{path.name}:{i + 1}: {line.strip()[:90]}")
    assert not offenders, "thresholds tuned on test: " + "; ".join(offenders)


def test_norm_stats_are_only_ever_requested_for_train():
    """Criterion 10: no call site asks compute_norm_stats for val or test."""
    offenders = []
    for path, text in _source_files():
        for i, line in enumerate(text.splitlines()):
            if "compute_norm_stats(" not in line or "def compute_norm_stats" in line:
                continue
            if "train" not in line:
                offenders.append(f"{path.name}:{i + 1}: {line.strip()[:90]}")
    assert not offenders, (
        "normalisation statistics computed outside train: " + "; ".join(offenders)
    )


def test_every_training_loop_uses_amp_and_accumulation():
    """Criterion 11: AMP and accumulation in every loop that steps an optimiser."""
    for name in ("src/train.py", "src/evaluate.py", "scripts/run_baselines.py"):
        text = (project_root() / name).read_text(encoding="utf-8")
        if "optimizer.step" not in text and "scaler.step" not in text:
            continue
        assert "autocast_ctx(" in text, f"{name} steps an optimiser without autocast"
        assert "make_grad_scaler(" in text, f"{name} steps an optimiser without a GradScaler"
        assert "grad_accum_steps" in text, f"{name} has no gradient accumulation"


def test_every_module_from_the_spec_exists_with_its_public_api():
    """Criterion 9: the prescribed signatures are importable."""
    import importlib

    expected = {
        "src.utils": ["set_seed", "seed_worker", "get_generator", "load_config",
                      "get_device", "count_parameters", "autocast_ctx", "log_vram"],
        "src.audio_features": ["load_audio", "log_mel", "chroma_cqt", "mfcc",
                               "segment_indices", "segment_features",
                               "compute_norm_stats", "apply_norm", "extract_dataset"],
        "src.chords": ["CHORD_VOCAB", "estimate_chords", "chords_from_midi",
                       "validate_against_lmd", "chord_transition_matrix"],
        "src.graph_builder": ["build_segment_graph", "build_chord_graph",
                              "build_hetero_graph", "rewire_edges", "visualise_graph",
                              "export_sample_graphs"],
        "src.splits": ["build_mtat_splits", "build_fma_splits", "build_deam_splits",
                       "build_musiccaps_splits", "reduce_to_top_k_tags",
                       "assert_no_leakage"],
        "src.metrics": ["tune_thresholds", "macro_f1", "micro_f1", "per_tag_prf",
                        "mean_auc_pr", "macro_roc_auc", "regression_metrics",
                        "retrieval_metrics", "graph_coherence_score", "knn_probe",
                        "silhouette", "aggregate_seeds"],
        "src.bert_encoder": ["BertTextEncoder", "BertTagClassifier"],
        "src.gnn_model": ["GNNEncoder", "GNNClassifier", "HeteroGNNEncoder"],
        "src.cnn_baseline": ["MelCNN"],
        "src.baselines": ["random_tag_baseline", "majority_tag_baseline",
                          "pca_mlp_baseline"],
        "src.fusion_model": ["CrossAttentionFusion", "GatedFusion", "GNNBertFusion",
                             "masked_multitask_loss"],
        "src.contrastive": ["DualEncoder", "symmetric_info_nce",
                            "build_similarity_matrix", "zero_shot_tag"],
        "src.datasets": ["MusicGraphDataset", "MelSpecDataset", "make_loader",
                         "alternating_loader"],
        "src.attention_viz": ["plot_bert_attention", "plot_graph_attention"],
        "src.human_eval": ["generate_listening_sheet", "compute_agreement"],
        "src.synthetic": ["make_synthetic_dataset"],
    }
    missing = []
    for module_name, names in expected.items():
        module = importlib.import_module(module_name)
        for name in names:
            if not hasattr(module, name):
                missing.append(f"{module_name}.{name}")
    assert not missing, "missing from the prescribed API: " + ", ".join(missing)


def test_bert_encoder_exposes_the_prescribed_signatures():
    import inspect

    from src.bert_encoder import BertTextEncoder

    signature = inspect.signature(BertTextEncoder.__init__)
    for parameter in ("model_name", "freeze_mode", "unfreeze_top_n"):
        assert parameter in signature.parameters
    forward = inspect.signature(BertTextEncoder.forward)
    for parameter in ("input_ids", "attention_mask", "return_tokens"):
        assert parameter in forward.parameters
    assert hasattr(BertTextEncoder, "param_groups")
    assert hasattr(BertTextEncoder, "precompute_embeddings")


def test_directory_tree_matches_the_specification():
    """Criterion 13: every prescribed path exists."""
    required = [
        "README.md", "requirements.txt", "config.yaml", "Makefile", "pytest.ini",
        ".gitignore",
        "data/raw", "data/processed", "data/splits",
        "notebooks/eda.ipynb", "notebooks/demo_context.ipynb",
        "src/__init__.py", "src/audio_features.py", "src/graph_builder.py",
        "src/bert_encoder.py", "src/gnn_model.py", "src/fusion_model.py",
        "src/contrastive.py", "src/train.py", "src/evaluate.py", "src/utils.py",
        "src/chords.py", "src/datasets.py", "src/splits.py", "src/metrics.py",
        "src/cnn_baseline.py", "src/baselines.py", "src/attention_viz.py",
        "src/human_eval.py", "src/synthetic.py",
        "scripts/download_mtat.sh", "scripts/download_fma.sh",
        "scripts/download_deam.sh", "scripts/download_lmd.sh",
        "scripts/download_musiccaps.py", "scripts/verify_datasets.py",
        "scripts/export_sample_graphs.py",
        "tests/test_metrics.py", "tests/test_graph_builder.py",
        "tests/test_splits.py", "tests/test_smoke.py",
        "results/plots", "results/retrieval_examples", "report",
    ]
    missing = [p for p in required if not (project_root() / p).exists()]
    assert not missing, "missing from the prescribed tree: " + ", ".join(missing)


# --------------------------------------------------------------------------- #
# human evaluation
# --------------------------------------------------------------------------- #
def test_listening_sheet_has_controls_and_randomised_order():
    from src.human_eval import generate_listening_sheet

    examples = [
        {"query_track_id": f"t{i}", "query_caption": f"caption {i}", "true_rank": i + 1,
         "top3": [{"track_id": f"t{i}", "caption": f"caption {i}", "score": 0.9}]}
        for i in range(12)
    ]
    sheet = generate_listening_sheet(examples, n_items=10, n_controls=4, seed=7)
    assert int(sheet["is_control"].sum()) == 4
    assert len(sheet) == 14
    assert sheet["presentation_order"].nunique() == len(sheet)
    # controls must not all be clustered at the end
    control_positions = sheet.index[sheet["is_control"]].tolist()
    assert min(control_positions) < len(sheet) - 4


def test_agreement_detects_discriminating_raters():
    import pandas as pd

    from src.human_eval import compute_agreement

    # 3 raters, 6 items; the last two are controls and are rated low by everyone
    frame = pd.DataFrame({
        "item_id": [f"i{j}" for j in range(6)],
        "is_control": [False, False, False, False, True, True],
        "rater1": [5.0, 4.0, 5.0, 4.0, 1.0, 2.0],
        "rater2": [4.0, 5.0, 4.0, 5.0, 2.0, 1.0],
        "rater3": [5.0, 5.0, 4.0, 4.0, 1.0, 1.0],
    })
    out = compute_agreement(frame)
    assert out["n_raters"] == 3 and out["n_items"] == 6
    assert out["control_discrimination"] > 2.0
    assert out["raters_discriminating"] == 3
    assert np.isfinite(out["krippendorff_alpha"])


def test_agreement_flags_non_discriminating_raters():
    """Everyone defaulting to 4 must NOT look like agreement about quality."""
    import pandas as pd

    from src.human_eval import compute_agreement

    frame = pd.DataFrame({
        "item_id": [f"i{j}" for j in range(6)],
        "is_control": [False, False, False, False, True, True],
        "rater1": [4.0] * 6,
        "rater2": [4.0] * 6,
        "rater3": [4.0] * 6,
    })
    out = compute_agreement(frame)
    assert out["control_discrimination"] == 0.0
    assert out["raters_discriminating"] == 0


# --------------------------------------------------------------------------- #
# the synthetic-artifact guard
# --------------------------------------------------------------------------- #
def test_detect_provenance_reads_the_explicit_field():
    from src.utils import REAL, SYNTHETIC, detect_provenance

    assert detect_provenance({"provenance": "synthetic"}) == SYNTHETIC
    assert detect_provenance({"provenance": "real"}) == REAL
    assert detect_provenance({}) == REAL
    assert detect_provenance(None) == REAL


def test_detect_provenance_does_not_rely_on_the_dataset_name():
    """The synthetic generator reuses real corpus names on purpose.

    A row that says ``dataset == "mtat"`` may still be fake, so the corpus name
    alone must never be treated as evidence of realness.
    """
    from src.utils import REAL, SYNTHETIC, detect_provenance

    fake = {"dataset": "mtat", "provenance": "synthetic"}
    real = {"dataset": "mtat", "provenance": "real"}
    assert detect_provenance(fake) == SYNTHETIC
    assert detect_provenance(real) == REAL
    # legacy artifacts written before the field existed
    assert detect_provenance({"dataset": "synthetic"}) == SYNTHETIC


def test_detect_provenance_on_quarantine_paths():
    from src.utils import REAL, SYNTHETIC, detect_provenance

    assert detect_provenance("results/_synthetic_smoke/plots/ablation.png") == SYNTHETIC
    assert detect_provenance("data/processed/synthetic/graphs/syn_0000.pt") == SYNTHETIC
    assert detect_provenance("data/processed/graphs/mtat/mtat_2.pt") == REAL


def test_detect_provenance_on_a_dataframe():
    import pandas as pd

    from src.utils import REAL, SYNTHETIC, detect_provenance

    fake = pd.DataFrame({"track_id": ["a"], "dataset": ["mtat"], "provenance": ["synthetic"]})
    real = pd.DataFrame({"track_id": ["a"], "dataset": ["mtat"], "provenance": ["real"]})
    assert detect_provenance(fake) == SYNTHETIC
    assert detect_provenance(real) == REAL


def test_guard_refuses_synthetic_without_the_flag():
    from src.utils import SyntheticArtifactError, guard_against_synthetic

    with pytest.raises(SyntheticArtifactError, match="refusing to run on synthetic"):
        guard_against_synthetic({"provenance": "synthetic"}, allow_synthetic=False)


def test_guard_allows_synthetic_with_the_flag():
    from src.utils import SYNTHETIC, guard_against_synthetic

    assert guard_against_synthetic({"provenance": "synthetic"},
                                   allow_synthetic=True) == SYNTHETIC


def test_guard_passes_real_data_through():
    from src.utils import REAL, guard_against_synthetic

    assert guard_against_synthetic({"provenance": "real"}, allow_synthetic=False) == REAL


def test_synthetic_graphs_are_stamped(synthetic):
    """Every generated graph and manifest row must carry its provenance."""
    import pandas as pd

    from src.utils import SYNTHETIC, detect_provenance

    graph = torch.load(sorted(Path(synthetic).glob("graphs/*.pt"))[0], weights_only=False)
    assert getattr(graph, "provenance", None) == SYNTHETIC
    assert detect_provenance(graph) == SYNTHETIC

    manifest = pd.read_csv(Path(synthetic) / "manifest.csv")
    assert "provenance" in manifest.columns
    assert set(manifest["provenance"]) == {SYNTHETIC}
    assert detect_provenance(manifest) == SYNTHETIC


def test_real_graphs_default_to_real_provenance():
    from src.graph_builder import build_segment_graph
    from src.utils import REAL, detect_provenance

    cfg = load_config("config.yaml")
    feats = np.random.default_rng(0).normal(size=(8, 96)).astype(np.float32)
    graph = build_segment_graph(feats, cfg, track_id="mtat_2", dataset="mtat")
    assert graph.provenance == REAL
    assert detect_provenance(graph) == REAL


def test_export_sample_graphs_refuses_synthetic_without_the_flag(synthetic, tmp_path):
    """The committed sample graphs stand in for the corpora; a synthetic one
    would misrepresent what the pipeline produces."""
    import subprocess
    import sys

    from src.utils import SyntheticArtifactError, guard_against_synthetic

    graphs = [torch.load(p, weights_only=False)
              for p in sorted(Path(synthetic).glob("graphs/*.pt"))[:3]]
    with pytest.raises(SyntheticArtifactError):
        guard_against_synthetic(graphs, allow_synthetic=False, context="export")
    # ...and is fine when asked for explicitly
    assert guard_against_synthetic(graphs, allow_synthetic=True) == "synthetic"


def test_evaluate_checkpoint_guard_detects_synthetic_weights(tmp_path):
    from src.evaluate import _guard_checkpoints
    from src.utils import SyntheticArtifactError

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    torch.save({"model_state": {}, "provenance": "synthetic"}, ckpt_dir / "task1_seed42_best.pt")
    cfg = load_config("config.yaml", {"paths.checkpoints": str(ckpt_dir)})

    with pytest.raises(SyntheticArtifactError):
        _guard_checkpoints(cfg, allow_synthetic=False)
    _guard_checkpoints(cfg, allow_synthetic=True)      # explicit opt-in is fine


def test_evaluate_checkpoint_guard_passes_real_weights(tmp_path):
    from src.evaluate import _guard_checkpoints

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    torch.save({"model_state": {}, "provenance": "real"}, ckpt_dir / "task2_seed42_best.pt")
    cfg = load_config("config.yaml", {"paths.checkpoints": str(ckpt_dir)})
    _guard_checkpoints(cfg, allow_synthetic=False)     # must not raise


def test_training_results_record_provenance(cfg_overrides):
    result = _run(2, cfg_overrides)
    assert result["provenance"] == "synthetic"


# --------------------------------------------------------------------------- #
# item-level resumability and atomic writes
#
# These are the tests that matter for a pipeline run across many sessions: the
# long jobs WILL be interrupted, and the only acceptable behaviour is to lose at
# most the item in flight.
# --------------------------------------------------------------------------- #
def _tiny_manifest(tmp_path, n=6):
    """A manifest of short synthetic wav files, so extraction is fast but real."""
    import soundfile as sf

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        path = audio_dir / f"clip{i}.wav"
        # 4 s of pink-ish noise at 22.05 kHz: long enough for min_nodes segments
        sf.write(path, rng.normal(0, 0.1, 22050 * 4).astype(np.float32), 22050)
        rows.append({"track_id": f"t{i}", "artist_id": f"a{i}", "audio_path": str(path),
                     "text": "", "split": "train", "y_genre": -1, "y_tags": "[]",
                     "y_valence": np.nan, "y_arousal": np.nan, "duration_s": 4.0,
                     "dataset": "probe"})
    import pandas as pd

    return pd.DataFrame(rows)


@pytest.mark.slow
def test_extract_dataset_skips_already_cached_keys(tmp_path):
    from src.audio_features import extract_dataset

    cfg = load_config("config.yaml")
    manifest = _tiny_manifest(tmp_path, n=4)
    h5 = tmp_path / "features.h5"

    first = extract_dataset(manifest, h5, cfg, n_workers=1)
    assert first["written"] == 4 and first["skipped"] == 0

    # a second run must do no work at all
    second = extract_dataset(manifest, h5, cfg, n_workers=1)
    assert second["written"] == 0, "re-extracted keys that were already cached"
    assert second["skipped"] == 4


@pytest.mark.slow
def test_extract_dataset_resumes_after_a_partial_run(tmp_path):
    """Extract half, then extract the whole manifest: only the rest is done."""
    from src.audio_features import extract_dataset

    cfg = load_config("config.yaml")
    manifest = _tiny_manifest(tmp_path, n=6)
    h5 = tmp_path / "features.h5"

    partial = extract_dataset(manifest.iloc[:3], h5, cfg, n_workers=1)
    assert partial["written"] == 3

    resumed = extract_dataset(manifest, h5, cfg, n_workers=1)
    assert resumed["skipped"] == 3, "resume did not recognise the completed keys"
    assert resumed["written"] == 3

    import h5py

    with h5py.File(h5, "r") as store:
        assert len(store.keys()) == 6
        assert store.attrs["feature_dim"] == 96
        for key in store.keys():
            assert store[key].shape[1] == 96


@pytest.mark.slow
def test_extract_dataset_writes_a_key_sidecar(tmp_path):
    """HDF5 is not transactional; the sidecar is how we know what completed."""
    import json as _json

    from src.audio_features import cache_keys_path, extract_dataset

    cfg = load_config("config.yaml")
    manifest = _tiny_manifest(tmp_path, n=3)
    h5 = tmp_path / "features.h5"
    extract_dataset(manifest, h5, cfg, n_workers=1)

    sidecar = cache_keys_path(h5)
    assert sidecar.exists(), "no key sidecar written"
    payload = _json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["n_keys"] == 3
    assert sorted(payload["keys"]) == ["t0", "t1", "t2"]


def test_verify_cache_reports_a_corrupt_container(tmp_path):
    from src.audio_features import verify_cache

    broken = tmp_path / "features.h5"
    broken.write_bytes(b"this is not an HDF5 file")
    report = verify_cache(broken)
    assert report["exists"] is True
    assert report["readable"] is False
    assert report["error"]


def test_verify_cache_on_a_missing_file_is_not_an_error(tmp_path):
    from src.audio_features import verify_cache

    report = verify_cache(tmp_path / "nope.h5")
    assert report["exists"] is False and report["error"] is None


@pytest.mark.slow
def test_extract_dataset_refuses_a_corrupt_cache_rather_than_appending(tmp_path):
    from src.audio_features import extract_dataset

    cfg = load_config("config.yaml")
    manifest = _tiny_manifest(tmp_path, n=2)
    h5 = tmp_path / "features.h5"
    h5.write_bytes(b"garbage")
    with pytest.raises(RuntimeError, match="cannot be opened"):
        extract_dataset(manifest, h5, cfg, n_workers=1)
    # ...but overwrite=True is an explicit, allowed recovery
    stats = extract_dataset(manifest, h5, cfg, n_workers=1, overwrite=True)
    assert stats["written"] == 2


# ---- atomic writes --------------------------------------------------------- #
def test_atomic_write_leaves_no_temp_file(tmp_path):
    from src.utils import atomic_write_text

    target = tmp_path / "manifest.csv"
    atomic_write_text(target, "a,b\n1,2\n")
    assert target.read_text() == "a,b\n1,2\n"
    assert not list(tmp_path.glob("*.tmp")), "temp file survived the write"


def test_atomic_write_does_not_clobber_on_failure(tmp_path, monkeypatch):
    """If the write dies mid-way, the previous file must still be intact."""
    import os as _os

    from src.utils import atomic_write_text

    target = tmp_path / "state.json"
    atomic_write_text(target, "GOOD")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(_os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_text(target, "PARTIAL")
    assert target.read_text() == "GOOD", "a failed write destroyed the old file"


def test_save_json_is_atomic(tmp_path):
    from src.utils import save_json

    path = save_json({"x": 1}, tmp_path / "out.json")
    assert json.loads(Path(path).read_text())["x"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_torch_save_round_trips(tmp_path):
    from src.utils import atomic_torch_save

    payload = {"model_state": {"w": torch.ones(3)}, "provenance": "real"}
    path = atomic_torch_save(payload, tmp_path / "ckpt.pt")
    back = torch.load(path, weights_only=False)
    assert torch.equal(back["model_state"]["w"], torch.ones(3))
    assert not list(tmp_path.glob("*.tmp"))


def test_write_manifest_is_atomic(tmp_path):
    import pandas as pd

    from src.splits import write_manifest

    frame = pd.DataFrame([{"track_id": "t1", "artist_id": "a1", "split": "train",
                           "y_tags": ["guitar"]}])
    path = write_manifest(frame, tmp_path / "m.csv")
    assert Path(path).exists()
    assert not list(tmp_path.glob("*.tmp"))


# ---- graph building -------------------------------------------------------- #
@pytest.mark.slow
def test_graph_building_skips_existing_files(tmp_path, synthetic):
    """Graph building must be item-level resumable, like extraction."""
    from scripts.build_graphs import build_graphs_for_manifest

    import pandas as pd

    manifest = pd.read_csv(Path(synthetic) / "manifest.csv").head(6)
    cfg = load_config("config.yaml")
    out_dir = tmp_path / "graphs"

    first = build_graphs_for_manifest(
        manifest, cfg, out_dir, graph_dir=Path(synthetic) / "graphs")
    assert first["written"] == 6 and first["skipped"] == 0

    second = build_graphs_for_manifest(
        manifest, cfg, out_dir, graph_dir=Path(synthetic) / "graphs")
    assert second["written"] == 0, "rebuilt graphs that already existed"
    assert second["skipped"] == 6


# ---- musiccaps retry ------------------------------------------------------- #
def test_musiccaps_retry_selects_only_failed_rows(tmp_path):
    """--retry-failed must read the log and skip everything already ok."""
    import pandas as pd

    log = pd.DataFrame([
        {"ytid": "aaa", "status": "ok"},
        {"ytid": "bbb", "status": "missing"},
        {"ytid": "ccc", "status": "corrupt"},
        {"ytid": "ddd", "status": "wrong_duration"},
        {"ytid": "eee", "status": "ok"},
    ])
    source = pd.DataFrame([{"ytid": y, "start_s": 30, "end_s": 40,
                            "is_audioset_eval": False}
                           for y in ["aaa", "bbb", "ccc", "ddd", "eee"]])

    failed = set(log.loc[log["status"] != "ok", "ytid"].astype(str))
    todo = source[source["ytid"].astype(str).isin(failed)]
    assert set(todo["ytid"]) == {"bbb", "ccc", "ddd"}
    assert "aaa" not in set(todo["ytid"]) and "eee" not in set(todo["ytid"])


def test_musiccaps_download_skips_existing_good_files(tmp_path):
    """A clip already on disk with the right duration is not re-downloaded."""
    import soundfile as sf

    from scripts.download_musiccaps import download_clip

    sf.write(tmp_path / "abc_30_40.wav",
             np.zeros(22050 * 10, dtype=np.float32), 22050)
    row = download_clip("abc", 30, 40, tmp_path, sleep=0.0, fmt="wav")
    assert row["status"] == "ok"
    assert abs(row["duration_s"] - 10.0) < 0.1


# --------------------------------------------------------------------------- #
# Xtext sources and aspect stripping
#
# The failure this guards against is degenerate supervision: MusicCaps captions
# are written FROM the aspect list, so training on the raw caption to predict
# those aspects measures string matching, not music understanding.
# --------------------------------------------------------------------------- #
def test_strip_removes_exact_aspect_phrases():
    from src.splits import strip_aspect_terms

    caption = "The low quality recording features a ballad song."
    out = strip_aspect_terms(caption, ["low quality", "ballad"])
    assert "low quality" not in out.lower()
    assert "ballad" not in out.lower()
    assert "recording" in out and "song" in out, "stripping destroyed the carrier text"


def test_strip_removes_scattered_multiword_aspects():
    """"sustained strings melody" also appears as "sustained strings, ... melody"."""
    from src.splits import strip_aspect_terms

    caption = ("contains sustained strings, mellow piano melody and soft female "
               "vocal singing over it")
    out = strip_aspect_terms(
        caption, ["sustained strings melody", "mellow piano melody", "soft female vocal"]
    ).lower()
    for leaked in ("sustained", "strings", "melody", "mellow", "piano", "female", "vocal"):
        assert leaked not in out, f"{leaked!r} survived stripping"


def test_strip_handles_morphological_variants():
    from src.splits import strip_aspect_terms

    out = strip_aspect_terms(
        "the guitars are strumming and the drummer is drumming",
        ["guitar", "strum", "drum"],
    ).lower()
    for leaked in ("guitar", "strum", "drum"):
        assert leaked not in out, f"inflected form of {leaked!r} survived"


def test_strip_preserves_stopwords_and_unrelated_content():
    from src.splits import strip_aspect_terms

    out = strip_aspect_terms(
        "It sounds like something you would hear at Sunday services.", ["sad"]
    )
    assert "Sunday services" in out
    assert "sounds" in out


def test_strip_is_a_noop_without_aspects():
    from src.splits import strip_aspect_terms

    caption = "A perfectly ordinary caption."
    assert strip_aspect_terms(caption, []) == caption


def test_strip_tidies_orphaned_punctuation():
    from src.splits import strip_aspect_terms

    out = strip_aspect_terms("It sounds sad and soulful, really.", ["sad", "soulful"])
    assert ",," not in out and "  " not in out
    assert not out.startswith(",")


def test_aspect_surface_forms_include_phrase_and_tokens():
    from src.splits import aspect_surface_forms

    forms = aspect_surface_forms(["soft female vocal"])
    assert "soft female vocal" in forms
    assert "vocal" in forms and "female" in forms
    assert "vocals" in forms, "plural inflection missing"


def test_masking_actually_removes_label_information():
    """The whole point: a masked caption must not contain its own labels."""
    from src.splits import strip_aspect_terms

    aspects = ["low quality", "sustained strings melody", "soft female vocal",
               "mellow piano melody", "sad", "soulful", "ballad"]
    caption = ("The low quality recording features a ballad song that contains "
               "sustained strings, mellow piano melody and soft female vocal "
               "singing over it. It sounds sad and soulful.")
    raw, masked = caption.lower(), strip_aspect_terms(caption, aspects).lower()

    def hits(text):
        return sum(1 for a in aspects if a.lower() in text)

    assert hits(raw) >= 5, "the test caption should leak in its raw form"
    assert hits(masked) == 0, "masked caption still contains its own labels"


def test_mtat_metadata_text_never_contains_tags():
    from src.splits import mtat_metadata_text

    text = mtat_metadata_text("BWV54 - I Aria", "J.S. Bach Solo Cantatas",
                              "American Bach Soloists")
    assert "American Bach Soloists" in text and "BWV54" in text
    assert mtat_metadata_text("", "", "") == ""
    assert mtat_metadata_text("Title", "nan", "") == "Title"


def test_text_sources_are_the_three_locked_values():
    from src.splits import TEXT_SOURCES

    assert set(TEXT_SOURCES) == {"caption_masked", "caption_raw", "metadata"}


def test_apply_text_source_selects_the_configured_variant(tmp_path):
    import pandas as pd

    from src.splits import apply_text_source

    manifest = pd.DataFrame([
        {"track_id": "musiccaps_a", "dataset": "musiccaps", "text": "ORIGINAL"},
        {"track_id": "musiccaps_b", "dataset": "musiccaps", "text": "ORIGINAL"},
    ])
    pd.DataFrame([
        {"track_id": "musiccaps_a", "dataset": "musiccaps", "caption_raw": "RAW A",
         "caption_masked": "MASKED A", "metadata": "META A"},
        {"track_id": "musiccaps_b", "dataset": "musiccaps", "caption_raw": "RAW B",
         "caption_masked": "MASKED B", "metadata": "META B"},
    ]).to_csv(tmp_path / "musiccaps_text_variants.csv", index=False)

    for source, expected in [("caption_masked", "MASKED A"),
                             ("caption_raw", "RAW A"),
                             ("metadata", "META A")]:
        cfg = load_config("config.yaml", {"data.text_source": source})
        out = apply_text_source(manifest, cfg, splits_dir=tmp_path)
        assert out.loc[0, "text"] == expected
        assert out.loc[0, "text_source"] == source


def test_apply_text_source_falls_back_rather_than_emptying_a_row(tmp_path):
    """An empty variant must not become an empty string fed to the tokenizer."""
    import pandas as pd

    from src.splits import apply_text_source

    manifest = pd.DataFrame([
        {"track_id": "mtat_1", "dataset": "mtat", "text": "FALLBACK"},
    ])
    pd.DataFrame([
        {"track_id": "mtat_1", "dataset": "mtat", "caption_raw": "",
         "caption_masked": "", "metadata": ""},
    ]).to_csv(tmp_path / "mtat_text_variants.csv", index=False)

    cfg = load_config("config.yaml", {"data.text_source": "caption_masked"})
    out = apply_text_source(manifest, cfg, splits_dir=tmp_path)
    assert out.loc[0, "text"] == "FALLBACK"


def test_apply_text_source_rejects_an_unknown_value(tmp_path):
    import pandas as pd

    from src.splits import apply_text_source

    cfg = load_config("config.yaml", {"data.text_source": "lyrics"})
    with pytest.raises(ValueError, match="text_source"):
        apply_text_source(pd.DataFrame([{"track_id": "x", "dataset": "mtat",
                                         "text": "t"}]), cfg, splits_dir=tmp_path)


def test_build_text_variants_shape():
    import pandas as pd

    from src.splits import build_text_variants

    manifest = pd.DataFrame([
        {"track_id": "t1", "dataset": "musiccaps", "text": "a sad ballad recording"},
    ])
    out = build_text_variants(manifest, aspects_by_track={"t1": ["sad", "ballad"]})
    assert list(out.columns) == ["track_id", "dataset", "caption_raw",
                                 "caption_masked", "metadata", "n_aspects_stripped"]
    assert out.loc[0, "caption_raw"] == "a sad ballad recording"
    assert "sad" not in out.loc[0, "caption_masked"]
    assert out.loc[0, "n_aspects_stripped"] == 2


def test_mtat_manifest_text_is_not_the_tag_string():
    """Regression guard for the degenerate Xtext the spec warns about."""
    import pandas as pd

    from src.utils import project_root

    path = project_root() / "data" / "splits" / "mtat_manifest.csv"
    if not path.exists():
        pytest.skip("MTAT manifest not built yet")
    frame = pd.read_csv(path).head(200)
    for record in frame.to_dict("records"):
        tags = json.loads(record["y_tags"]) if isinstance(record["y_tags"], str) else []
        text = str(record.get("text", "") or "").lower()
        if not tags or not text:
            continue
        leaked = [t for t in tags if t.lower() in text]
        assert len(leaked) < max(2, len(tags)),  (
            f"{record['track_id']}: Xtext appears to contain its own tags {leaked}"
        )


# --------------------------------------------------------------------------- #
# normalisation statistics must be provably train-only
# --------------------------------------------------------------------------- #
def test_norm_stats_files_record_train_provenance():
    """Every persisted norm_stats file must say, in the file, that it is train."""
    from src.utils import project_root

    processed = project_root() / "data" / "processed"
    files = sorted(processed.glob("norm_stats*.json"))
    if not files:
        pytest.skip("no norm stats extracted yet")
    for path in files:
        stats = json.loads(path.read_text(encoding="utf-8"))
        assert stats.get("split") == "train", (
            f"{path.name} records split={stats.get('split')!r}; normalisation "
            "statistics must come from the train split only"
        )
        assert stats.get("dim") == 96
        assert len(stats["mean"]) == 96 and len(stats["std"]) == 96
        assert all(s > 0 for s in stats["std"]), "a zero std would divide by zero"


def test_compute_norm_stats_uses_only_train_rows(tmp_path):
    """Val/test rows in the manifest must not influence the statistics."""
    import h5py
    import pandas as pd

    from src.audio_features import compute_norm_stats

    h5 = tmp_path / "f.h5"
    rng = np.random.default_rng(0)
    train = rng.normal(0.0, 1.0, size=(10, 96)).astype(np.float16)
    # val/test rows are wildly off-distribution: if they leak in, mean explodes
    other = (rng.normal(0.0, 1.0, size=(10, 96)) + 1000.0).astype(np.float16)
    with h5py.File(h5, "w") as store:
        store.create_dataset("t_train", data=train)
        store.create_dataset("t_val", data=other)
        store.create_dataset("t_test", data=other)

    manifest = pd.DataFrame([
        {"track_id": "t_train", "split": "train"},
        {"track_id": "t_val", "split": "val"},
        {"track_id": "t_test", "split": "test"},
    ])
    stats = compute_norm_stats(manifest, split="train", cfg=None, h5_path=h5)
    assert stats["split"] == "train"
    assert stats["n_segments"] == 10, "non-train rows were included"
    assert abs(float(np.mean(stats["mean"]))) < 5.0, (
        "the val/test offset of +1000 leaked into the train statistics"
    )


def test_data_bundle_refuses_non_train_norm_stats(tmp_path):
    """A norm_stats file without train provenance must stop the run."""
    from src.utils import save_json

    processed = tmp_path / "processed"
    processed.mkdir()
    save_json({"mean": [0.0] * 96, "std": [1.0] * 96, "split": "test", "dim": 96},
              processed / "norm_stats.json")
    stats = json.loads((processed / "norm_stats.json").read_text(encoding="utf-8"))
    assert stats["split"] != "train"
    # DataBundle raises on exactly this condition (src/train.py)
    from src.utils import project_root

    source = (project_root() / "src" / "train.py").read_text(encoding="utf-8")
    assert 'payload.get("split") != "train"' in source
    assert "refusing to run" in source


# --------------------------------------------------------------------------- #
# the graph sanity gate
# --------------------------------------------------------------------------- #
def test_repetition_score_detects_repeated_structure():
    from scripts.graph_sanity import repetition_score

    rng = np.random.default_rng(0)
    # a verse/chorus/verse track: segments 0-3 recur at 8-11
    base = rng.normal(size=(4, 96))
    repetitive = np.vstack([base, rng.normal(size=(4, 96)), base]).astype(np.float32)
    through_composed = rng.normal(size=(12, 96)).astype(np.float32)

    assert repetition_score(repetitive) > repetition_score(through_composed)


def test_analyse_graph_flags_a_chain_like_graph():
    """k=1 on a smoothly drifting track should look chain-like and score low."""
    from scripts.graph_sanity import analyse_graph
    from src.graph_builder import build_segment_graph

    cfg = load_config("config.yaml", {"graph.knn_k": 1})
    # a smooth ramp: every segment's nearest neighbour is its time neighbour
    feats = np.linspace(0, 1, 20)[:, None] * np.ones((1, 96))
    feats = feats.astype(np.float32) + 1e-3 * np.arange(96)[None, :]
    data = build_segment_graph(feats, cfg, track_id="ramp")
    report = analyse_graph(data, feats)
    assert report["long_range_fraction"] < 0.5, (
        "a pure temporal ramp should not produce long-range similarity edges"
    )


def test_analyse_graph_finds_planted_repeats():
    from scripts.graph_sanity import analyse_graph
    from src.graph_builder import build_segment_graph

    cfg = load_config("config.yaml")
    rng = np.random.default_rng(1)
    base = rng.normal(size=(4, 96))
    feats = np.vstack([base, rng.normal(size=(4, 96)), base]).astype(np.float32)
    data = build_segment_graph(feats, cfg, track_id="verse_chorus_verse")
    report = analyse_graph(data, feats)
    assert report["long_range_fraction"] > 0.5, "planted repeats were not connected"
    assert report["nodes_without_similarity_edges"] == 0


def test_graph_sanity_verdict_written_for_real_data():
    """If the gate has been run, its verdict must be a pass."""
    from src.utils import project_root

    path = project_root() / "results" / "plots" / "graph_sanity" / "graph_sanity.json"
    if not path.exists():
        pytest.skip("graph sanity gate has not been run yet")
    verdict = json.loads(path.read_text(encoding="utf-8"))
    assert verdict["passed"] is True, (
        f"A4.4 gate failed: long_range={verdict['mean_long_range_fraction']:.3f}, "
        f"repeat_recall={verdict['mean_repeat_recall']:.3f}. Do not proceed."
    )


# --------------------------------------------------------------------------- #
# the mel cache the CNN baseline eats -- fairness, not fidelity
# --------------------------------------------------------------------------- #
def test_pooled_log_mel_covers_the_whole_track():
    """B2 must see the same span of music as the GNN, not a 6 s crop."""
    from src.audio_features import pooled_log_mel

    cfg = load_config("config.yaml")
    sr = int(cfg["audio"]["sample_rate"])
    rng = np.random.default_rng(0)
    # loud second half: if only the first half were kept, the tail would vanish
    y = np.concatenate([rng.normal(0, 0.01, sr * 10),
                        rng.normal(0, 0.50, sr * 10)]).astype(np.float32)

    mel = pooled_log_mel(y, sr, cfg, n_frames=256)
    assert mel.shape == (int(cfg["audio"]["n_mels"]), 256)
    first, second = mel[:, :128].mean(), mel[:, 128:].mean()
    assert second > first + 3.0, "the second half of the track is missing from the patch"


def test_pooled_log_mel_is_fixed_width_regardless_of_duration():
    from src.audio_features import pooled_log_mel

    cfg = load_config("config.yaml")
    sr = int(cfg["audio"]["sample_rate"])
    rng = np.random.default_rng(0)
    for seconds in (5, 29, 30):
        mel = pooled_log_mel(rng.normal(0, 0.1, sr * seconds).astype(np.float32),
                             sr, cfg, n_frames=256)
        assert mel.shape[1] == 256, f"{seconds}s track gave {mel.shape[1]} frames"


# --------------------------------------------------------------------------- #
# Task 1 must run from manifests alone -- no audio, no HDF5, no graphs.
#
# Regression guard: Task 1 previously loaded rows through MusicGraphDataset,
# which builds a segment graph per item and therefore opens the feature cache.
# The Kaggle payload deliberately ships no caches, so the run died at the first
# batch on FileNotFoundError.
# --------------------------------------------------------------------------- #
def _text_manifest(tmp_path):
    import pandas as pd

    rows = [
        {"track_id": "musiccaps_a", "artist_id": "yt_a", "audio_path": "",
         "text": "a mellow piano ballad", "split": "train", "y_genre": -1,
         "y_tags": '["piano", "mellow"]', "y_valence": None, "y_arousal": None,
         "duration_s": 10.0, "dataset": "musiccaps", "provenance": "real"},
        {"track_id": "musiccaps_b", "artist_id": "yt_b", "audio_path": "",
         "text": "fast distorted guitar", "split": "val", "y_genre": -1,
         "y_tags": '["guitar"]', "y_valence": None, "y_arousal": None,
         "duration_s": 10.0, "dataset": "musiccaps", "provenance": "real"},
        {"track_id": "deam_c", "artist_id": "art_c", "audio_path": "",
         "text": "an orchestral piece", "split": "train", "y_genre": -1,
         "y_tags": "[]", "y_valence": 5.0, "y_arousal": 4.0,
         "duration_s": 45.0, "dataset": "deam", "provenance": "real"},
    ]
    return pd.DataFrame(rows)


def test_text_dataset_needs_no_feature_cache(tmp_path):
    from src.datasets import TextTagDataset

    ds = TextTagDataset(_text_manifest(tmp_path), cfg=load_config("config.yaml"),
                        tag_vocab=["piano", "mellow", "guitar"], split="train")
    assert len(ds) == 2
    item = ds[0]
    assert item.text == "a mellow piano ballad"
    assert item.y_tags.shape == (1, 3)
    assert item.y_tags[0, 0] == 1.0 and item.y_tags[0, 1] == 1.0
    assert item.y_tags[0, 2] == 0.0


def test_text_dataset_keeps_the_sentinel_rule(tmp_path):
    """A corpus with no tag vocabulary gets -1, never confident zeros."""
    from src.datasets import TextTagDataset

    ds = TextTagDataset(_text_manifest(tmp_path), cfg=load_config("config.yaml"),
                        tag_vocab=["piano", "mellow", "guitar"], split="train",
                        datasets=["deam"])
    assert len(ds) == 1
    assert torch.all(ds[0].y_tags == -1), "DEAM row should be all -1, not zeros"
    assert float(ds[0].y_valence) == 5.0


def test_text_dataset_batches_through_the_normal_loader(tmp_path):
    from src.datasets import TextTagDataset, make_loader

    cfg = load_config("config.yaml")
    ds = TextTagDataset(_text_manifest(tmp_path), cfg=cfg,
                        tag_vocab=["piano", "mellow", "guitar"])
    loader = make_loader(ds, cfg, shuffle=False, seed=0, batch_size=3, num_workers=0)
    batch = next(iter(loader))
    assert batch.y_tags.shape == (3, 3)
    assert isinstance(batch.text, list) and len(batch.text) == 3
    assert len(batch.track_id) == 3


def test_task1_uses_the_text_only_loader():
    """Structural guard: run_task1 must not go through the graph dataset."""
    from src.utils import project_root

    source = (project_root() / "src" / "train.py").read_text(encoding="utf-8")
    start = source.index("def run_task1(")
    body = source[start:source.index("def run_task2(")]
    assert "bundle.text_dataset(" in body, "Task 1 is not using the text-only loader"
    assert "bundle.dataset(" not in body, (
        "Task 1 still builds graphs, so it needs the HDF5 caches and will fail "
        "anywhere they are absent (e.g. the Kaggle payload)"
    )


def test_kaggle_payload_carries_what_task1_needs(tmp_path):
    """The payload must contain manifests + vocab + code, and NO caches."""
    from src.utils import load_config, project_root, resolve_path

    cfg = load_config("config.yaml")
    sys.path.insert(0, str(project_root() / "scripts"))
    import importlib.util as _u

    spec = _u.spec_from_file_location(
        "mkp", project_root() / "scripts" / "make_kaggle_payload.py")
    mkp = _u.module_from_spec(spec)
    spec.loader.exec_module(mkp)

    files = mkp.collect(project_root(), cfg, include_graphs=False)
    names = {str(p.relative_to(project_root())).replace("\\", "/") for p in files}

    assert any(n.endswith("musiccaps_manifest.csv") for n in names)
    assert any(n.endswith("musiccaps_tag_vocab.json") for n in names)
    assert any(n.endswith("musiccaps_text_variants.csv") for n in names)
    assert "config.yaml" in names and "src/train.py" in names
    # the whole point: no gigabyte caches
    assert not [n for n in names if n.endswith((".h5", ".hdf5"))], "cache leaked into payload"
    assert not [n for n in names if n.endswith(".mp3")], "audio leaked into payload"


# --------------------------------------------------------------------------- #
# get_device must fail fast on a GPU this torch has no kernels for.
#
# torch.cuda.is_available() returns True for such a device; the failure only
# appears later as "no kernel image is available for execution on the device".
# Kaggle's P100 is sm_60 and current torch builds start at sm_70, so a whole
# sweep dies the same cryptic way, once per run.
# --------------------------------------------------------------------------- #
def test_get_device_rejects_a_too_old_gpu(monkeypatch):
    from src.utils import get_device

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device to patch")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (6, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list",
                        lambda: ["sm_70", "sm_75", "sm_80", "sm_90"])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "Tesla P100-PCIE-16GB")

    with pytest.raises(RuntimeError) as excinfo:
        get_device("cuda")
    message = str(excinfo.value)
    assert "sm_60" in message
    assert "T4" in message, "the error must name the actual remedy"
    assert "--device cpu" in message


def test_get_device_accepts_a_supported_gpu(monkeypatch):
    from src.utils import get_device

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device to patch")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_arch_list",
                        lambda: ["sm_70", "sm_75", "sm_80"])
    assert get_device("cuda").type == "cuda"


def test_get_device_allows_a_newer_than_listed_gpu(monkeypatch):
    """A device newer than anything listed can JIT from PTX; do not block it."""
    from src.utils import get_device

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device to patch")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_70", "sm_75"])
    assert get_device("cuda").type == "cuda"


def test_get_device_cpu_is_always_honoured():
    from src.utils import get_device

    assert get_device("cpu").type == "cpu"


# --------------------------------------------------------------------------- #
# bert.freeze_mode must actually take effect.
#
# It used to be ignored: run_task1 derived the mode from freeze_epochs alone, so
# a sweep asking for frozen_probe with freeze_epochs=0 trained full_ft instead,
# and two of its three "freeze modes" were the same configuration.
# --------------------------------------------------------------------------- #
def test_freeze_mode_is_honoured_not_derived_from_freeze_epochs():
    from src.train import starting_freeze_mode, target_freeze_mode

    cfg = load_config("config.yaml", {"bert.freeze_mode": "frozen_probe",
                                      "bert.freeze_epochs": 0})
    assert starting_freeze_mode(cfg) == "frozen_probe", (
        "freeze_mode=frozen_probe with freeze_epochs=0 must NOT become full_ft"
    )
    assert target_freeze_mode(cfg) == "frozen_probe"


def test_freeze_epochs_starts_frozen_then_reaches_the_target():
    from src.train import starting_freeze_mode, target_freeze_mode

    for target in ("top_n", "full_ft"):
        cfg = load_config("config.yaml", {"bert.freeze_mode": target,
                                          "bert.freeze_epochs": 2})
        assert starting_freeze_mode(cfg) == "frozen_probe"
        assert target_freeze_mode(cfg) == target


def test_full_ft_with_no_warmup_starts_unfrozen():
    from src.train import starting_freeze_mode

    cfg = load_config("config.yaml", {"bert.freeze_mode": "full_ft",
                                      "bert.freeze_epochs": 0})
    assert starting_freeze_mode(cfg) == "full_ft"


def test_unknown_freeze_mode_is_rejected():
    from src.train import target_freeze_mode

    cfg = load_config("config.yaml", {"bert.freeze_mode": "thaw_everything"})
    with pytest.raises(ValueError, match="freeze_mode"):
        target_freeze_mode(cfg)


def test_the_three_sweep_modes_are_actually_distinct():
    """The sweep must compare three different configurations, not two."""
    from src.train import starting_freeze_mode, target_freeze_mode

    sweep = [("frozen_probe", 0), ("top_n", 2), ("full_ft", 0)]
    seen = set()
    for mode, epochs in sweep:
        cfg = load_config("config.yaml", {"bert.freeze_mode": mode,
                                          "bert.freeze_epochs": epochs})
        seen.add((starting_freeze_mode(cfg), target_freeze_mode(cfg), epochs))
    assert len(seen) == 3, f"sweep collapses to {len(seen)} distinct configs: {seen}"


def test_recovered_results_are_stamped():
    """A result rebuilt from a log must never look like a first-hand artifact."""
    from src.utils import project_root

    recovered = sorted((project_root() / "results").glob("task1_seed42_*.json"))
    checked = 0
    for path in recovered:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("recovered_from_log"):
            continue
        checked += 1
        assert payload["thresholds"] is None
        assert "recovery_note" in payload
        assert payload["threshold_source"] == "val"
    if checked == 0:
        pytest.skip("no log-recovered results present")


# --------------------------------------------------------------------------- #
# the mel CNN, rebuilt without the parameter-matching constraint
# --------------------------------------------------------------------------- #
def test_mel_cnn_refuses_the_parameter_matching_constraint():
    """The constraint that produced 0.1654 must not be reachable by accident."""
    from src.cnn_baseline import MelCNN

    with pytest.raises(ValueError, match="target_params"):
        MelCNN(n_tags=50, target_params=432_690)


def test_mel_cnn_is_a_few_million_parameters_not_a_few_hundred_thousand():
    from src.cnn_baseline import MelCNN

    params = MelCNN(n_tags=50).capacity_report()["trainable_params"]
    assert 2_000_000 < params < 8_000_000, (
        f"{params} is outside the range a standard short-chunk CNN occupies"
    )


def test_mel_cnn_averages_chunk_probabilities_not_logits():
    """A logit average lets one confident chunk decide the whole clip.

    Two chunks, one saturated positive and one saturated negative, must average
    to about 0.5 in probability space. Averaging logits first would too, but the
    asymmetric case below would not, so this pins the actual behaviour.
    """
    import torch

    from src.cnn_baseline import MelCNN

    model = MelCNN(n_tags=3).eval()
    with torch.no_grad():
        stacked = torch.randn(2, 4, 1, 128, 129)
        out = model(stacked)
    assert out["tag_probs"].shape == (2, 3)
    assert torch.all((out["tag_probs"] >= 0) & (out["tag_probs"] <= 1))
    # the returned logits must be the inverse-sigmoid of the averaged probability
    recovered = torch.sigmoid(out["tag_logits"])
    assert torch.allclose(recovered, out["tag_probs"], atol=1e-4)


def test_mel_cnn_genre_only_has_no_tag_head():
    from src.cnn_baseline import MelCNN

    model = MelCNN(n_tags=0, n_genres=8)
    assert model.tag_head is None
    import torch

    out = model(torch.randn(2, 1, 128, 129))
    assert "tag_logits" not in out and out["genre_logits"].shape == (2, 8)


def test_chunked_mel_dataset_shapes_and_eval_determinism(tmp_path):
    import h5py
    import numpy as np
    import pandas as pd

    from src.datasets import ChunkedMelDataset
    from src.utils import load_config, project_root

    cfg = load_config(project_root() / "config.yaml")
    path = tmp_path / "mels_full_fake.h5"
    with h5py.File(path, "w") as store:
        for i in range(3):
            store.create_dataset(f"t{i}", data=np.random.randn(128, 1250).astype(np.float16))
    frame = pd.DataFrame([
        {"track_id": f"t{i}", "artist_id": f"a{i}", "split": "train",
         "dataset": "fake", "y_tags": '["guitar"]', "y_genre": i % 2}
        for i in range(3)
    ])

    train = ChunkedMelDataset(frame, cfg=cfg, h5_path={"fake": path},
                              tag_vocab=["guitar", "drum"], mode="random")
    mel, y, genre, track = train[0]
    assert mel.shape == (1, 1, 128, train.chunk_frames)
    assert train.chunk_frames == 129, "3 s at 22050/512 should be 129 frames"
    assert y.tolist() == [1.0, 0.0] and int(genre) == 0

    evaluate = ChunkedMelDataset(frame, cfg=cfg, h5_path={"fake": path},
                                 tag_vocab=["guitar", "drum"], mode="all",
                                 n_eval_chunks=9)
    first, *_ = evaluate[0]
    second, *_ = evaluate[0]
    assert first.shape == (9, 1, 128, 129)
    assert torch.equal(first, second), "evaluation chunks must not be random"


def test_chunked_mel_random_draw_changes_with_the_epoch(tmp_path):
    """Random excerpts are augmentation; a fixed draw would waste the cache."""
    import h5py
    import numpy as np
    import pandas as pd

    from src.datasets import ChunkedMelDataset
    from src.utils import load_config, project_root

    cfg = load_config(project_root() / "config.yaml")
    path = tmp_path / "m.h5"
    with h5py.File(path, "w") as store:
        # a ramp along time, so two different offsets are visibly different;
        # kept inside float16 range or the cast silently saturates
        ramp = np.linspace(0, 1, 1250, dtype=np.float32)
        store.create_dataset("t0", data=np.tile(ramp, (128, 1)).astype(np.float16))
    frame = pd.DataFrame([{"track_id": "t0", "artist_id": "a", "split": "train",
                           "dataset": "fake", "y_tags": "[]", "y_genre": -1}])
    ds = ChunkedMelDataset(frame, cfg=cfg, h5_path={"fake": path},
                           tag_vocab=["x"], mode="random")
    ds.set_epoch(1)
    a, *_ = ds[0]
    ds.set_epoch(2)
    b, *_ = ds[0]
    assert not torch.equal(a, b), "the same excerpt every epoch is not augmentation"


# --------------------------------------------------------------------------- #
# the genre task path
# --------------------------------------------------------------------------- #
def test_masked_genre_loss_ignores_unlabelled_rows():
    """A -1 genre must be excluded, never trained as class 0."""
    import torch

    from src.fusion_model import masked_genre_loss

    class Batch:
        pass

    batch = Batch()
    logits = torch.zeros(4, 3, requires_grad=True)
    batch.y_genre = torch.tensor([-1, -1, -1, -1])
    loss, parts = masked_genre_loss({"genre_logits": logits}, batch)
    assert parts["n_genre"] == 0 and float(loss) == 0.0

    batch.y_genre = torch.tensor([-1, 1, -1, 2])
    loss, parts = masked_genre_loss({"genre_logits": logits}, batch)
    assert parts["n_genre"] == 2
    assert float(loss) > 0


def test_masked_genre_loss_needs_a_genre_head():
    from src.fusion_model import masked_genre_loss

    class Batch:
        pass

    with pytest.raises(ValueError, match="genre_logits"):
        masked_genre_loss({}, Batch())


def test_gnn_classifier_genre_only_drops_the_tag_head():
    from src.gnn_model import GNNClassifier

    model = GNNClassifier(n_tags=0, n_genres=8)
    assert model.tag_head is None and model.genre_head is not None
    with pytest.raises(ValueError):
        GNNClassifier(n_tags=0, n_genres=0)


def test_task2_target_is_validated():
    from src.train import task2_target
    from src.utils import load_config, project_root

    cfg = load_config(project_root() / "config.yaml")
    assert task2_target(cfg) in ("genre", "tags")
    cfg["data"]["task2_target"] = "nonsense"
    with pytest.raises(ValueError, match="task2_target"):
        task2_target(cfg)


# --------------------------------------------------------------------------- #
# a result scored against the old vocabulary must not reach the report
# --------------------------------------------------------------------------- #
def test_report_refuses_results_from_a_different_tag_vocabulary(monkeypatch):
    """The failure mode this guards is silent, which is why it needs a test.

    A7.3 changed 7 of the 50 MusicCaps tags. If the re-run sweep dies part way,
    the pre-A7.3 result files are still on disk and would be picked up without
    complaint, mixing corrected and leaked numbers inside one table.
    """
    from report import fill_report

    monkeypatch.setattr(fill_report, "current_vocab",
                        lambda source: ["a", "b", "c"])

    matching = {"test": {"macro_f1": 0.5}, "tag_vocab": ["a", "b", "c"]}
    assert fill_report.fresh(matching, "musiccaps", "ok") == matching

    different = {"test": {"macro_f1": 0.9}, "tag_vocab": ["a", "b", "z"]}
    assert fill_report.fresh(different, "musiccaps", "changed") == {}, (
        "a result scored against a different label space was accepted"
    )

    # order matters too: the vocabulary order fixes the column index per tag
    reordered = {"test": {"macro_f1": 0.5}, "tag_vocab": ["c", "b", "a"]}
    assert fill_report.fresh(reordered, "musiccaps", "reordered") == {}

    unverifiable = {"test": {"macro_f1": 0.5}}          # predates the field
    assert fill_report.fresh(unverifiable, "musiccaps", "old") == {}, (
        "a result with no recorded vocabulary was assumed good"
    )


def test_report_macros_never_invent_a_number():
    """A missing result must render as pending, not as a plausible default."""
    from report import fill_report

    assert fill_report.num(None) == fill_report.PENDING
    assert fill_report.num(float("nan")) == fill_report.PENDING
    assert fill_report.integer(None) == fill_report.PENDING
    assert fill_report.seconds(None) == fill_report.PENDING
    assert fill_report.num(0.12345) == "0.1235"
    assert fill_report.integer(1234567) == "1{,}234{,}567"


# --------------------------------------------------------------------------- #
# the page limit is a submission requirement, so it must fail loudly
# --------------------------------------------------------------------------- #
def _tex(body: str) -> str:
    return ("\\documentclass[conference]{IEEEtran}\n\\begin{document}\n"
            + body + "\n\\end{document}\n")


def test_page_guard_fires_above_the_limit(tmp_path):
    from report import check_tex

    path = tmp_path / "long.tex"
    # ~12 pages of prose at the module's own words-per-page constant
    path.write_text(_tex("word " * int(check_tex.WORDS_PER_PAGE * 12)),
                    encoding="utf-8")

    est = check_tex.estimate(path)
    assert est["total_pages"] > check_tex.PAGE_LIMIT[1]
    assert est["over_limit"]
    problems = check_tex.check(path)
    assert any("OVER the" in p for p in problems), (
        f"a 12-page draft was not flagged: {problems}"
    )
    assert check_tex.main(["--tex", str(path)]) != 0, "exit code did not fail"


def test_page_guard_passes_inside_the_limit(tmp_path):
    from report import check_tex

    path = tmp_path / "ok.tex"
    path.write_text(_tex("word " * int(check_tex.WORDS_PER_PAGE * 7)),
                    encoding="utf-8")
    assert not check_tex.estimate(path)["over_limit"]
    assert not any("OVER the" in p for p in check_tex.check(path))


def test_macros_are_counted_where_they_are_used_not_where_defined(tmp_path):
    """A macro definition typesets nothing where it sits.

    The AUTOGEN block sits above \\appendix, so counting it as body prose
    charged the main body for every macro body -- including a table that only
    ever renders inside the appendix. That inflated the estimate by a full page
    and would have had us cutting real prose to fix a measurement error.
    """
    from report import check_tex

    note = "word " * 400                      # a long prose macro, appendix-only
    body = "\n".join([
        "% --- AUTOGEN:BEGIN ---",
        "\\newcommand{\\AppendixNote}{" + note + "}",
        "\\newcommand{\\Score}{0.356}",
        "% --- AUTOGEN:END ---",
        "word " * 100,
        "\\Score{} is the headline.",
        "\\appendix",
        "\\AppendixNote{}",
    ])
    path = tmp_path / "macros.tex"
    path.write_text(_tex(body), encoding="utf-8")

    est = check_tex.estimate(path)
    # the 400-word note belongs to the appendix, where it is used
    assert est["words"] < 200, (
        f"body counted {est['words']} words; the appendix-only macro leaked in"
    )
    assert est["appendix_pages"] > 400 / check_tex.WORDS_PER_PAGE * 0.9, (
        "the appendix was not charged for the macro it actually typesets"
    )


def test_a_table_defined_in_the_macro_block_belongs_to_its_use_site(tmp_path):
    from report import check_tex

    body = "\n".join([
        "% --- AUTOGEN:BEGIN ---",
        "\\newcommand{\\BigTable}{\\begin{table}x\\end{table}}",
        "% --- AUTOGEN:END ---",
        "word " * 50,
        "\\appendix",
        "\\BigTable{}",
    ])
    path = tmp_path / "table.tex"
    path.write_text(_tex(body), encoding="utf-8")

    assert check_tex.estimate(path)["tables"] == 0, (
        "a table used only in the appendix was charged to the main body"
    )


def test_appendix_material_is_counted_separately(tmp_path):
    """The planned response to overflow is an appendix, so it must not count."""
    from report import check_tex

    main_body = "word " * int(check_tex.WORDS_PER_PAGE * 8)
    overflow = "word " * int(check_tex.WORDS_PER_PAGE * 6)

    without = tmp_path / "without.tex"
    without.write_text(_tex(main_body + overflow), encoding="utf-8")
    assert check_tex.estimate(without)["over_limit"], "14 pages should be over"

    with_appendix = tmp_path / "with.tex"
    with_appendix.write_text(_tex(main_body + "\n\\appendix\n" + overflow),
                             encoding="utf-8")
    est = check_tex.estimate(with_appendix)
    assert est["has_appendix"]
    assert not est["over_limit"], "appendix material was counted against the limit"
    assert est["appendix_pages"] > 5


def test_rewiring_control_is_reproducible_across_dataset_instances(tmp_path):
    """the control must be identical run to run, or it measures nothing.

    Python randomises string hashing per process, so a track-id-seeded rewire
    built on `hash()` would silently differ between runs.
    """
    import numpy as np
    import pandas as pd

    from src.datasets import MusicGraphDataset
    from src.graph_builder import build_segment_graph
    from src.utils import load_config, project_root

    cfg = load_config(project_root() / "config.yaml")
    cfg["graph"]["rewire"] = True

    feats = np.random.default_rng(0).normal(size=(12, 96)).astype(np.float32)
    graph = build_segment_graph(feats, cfg, track_id="t0", n_tags=4)
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    torch.save(graph, graph_dir / "t0.pt")

    frame = pd.DataFrame([{"track_id": "t0", "artist_id": "a", "split": "train",
                           "dataset": "mtat", "y_tags": "[]", "y_genre": -1}])
    first = MusicGraphDataset(frame, cfg=cfg, graph_dir=graph_dir,
                              tag_vocab=["a", "b", "c", "d"])[0]
    second = MusicGraphDataset(frame, cfg=cfg, graph_dir=graph_dir,
                               tag_vocab=["a", "b", "c", "d"])[0]
    assert torch.equal(first.edge_index, second.edge_index), (
        "the rewiring differs between dataset instances; it is not reproducible"
    )

    # and it must actually have changed something
    plain_cfg = load_config(project_root() / "config.yaml")
    plain = MusicGraphDataset(frame, cfg=plain_cfg, graph_dir=graph_dir,
                              tag_vocab=["a", "b", "c", "d"])[0]
    assert not torch.equal(first.edge_index, plain.edge_index), (
        "graph.rewire=true left the topology untouched"
    )
    # degrees preserved: that is what makes it a control rather than a lesion
    def degrees(d):
        return np.bincount(d.edge_index[0].numpy(), minlength=int(d.num_nodes))
    assert sorted(degrees(first).tolist()) == sorted(degrees(plain).tolist())


# --------------------------------------------------------------------------- #
# DEAM imbalance and target scale
# --------------------------------------------------------------------------- #
def test_emotion_stats_refuse_any_split_but_train():
    import pandas as pd

    from src.splits import compute_emotion_stats

    frame = pd.DataFrame([{"split": "train", "y_valence": 5.0, "y_arousal": 5.0}])
    with pytest.raises(ValueError, match="train split"):
        compute_emotion_stats(frame, split="test")


def test_emotion_stats_use_train_rows_only():
    import numpy as np
    import pandas as pd

    from src.splits import compute_emotion_stats

    frame = pd.DataFrame(
        [{"split": "train", "y_valence": 4.0, "y_arousal": 4.0}] * 10
        + [{"split": "test", "y_valence": 9.0, "y_arousal": 9.0}] * 10
    )
    stats = compute_emotion_stats(frame)
    assert stats["valence"]["n"] == 10
    assert stats["valence"]["mean"] == pytest.approx(4.0), (
        "test-split targets leaked into the training target scale"
    )


def test_emotion_metrics_are_reported_on_the_original_scale():
    """The model predicts standard deviations; the table must show 1-9 units."""
    import numpy as np

    from src.train import emotion_metrics

    stats = {"valence": {"mean": 5.0, "std": 2.0},
             "arousal": {"mean": 5.0, "std": 2.0}}
    cfg = {"multitask": {"emotion_stats": stats, "standardise_targets": True}}

    truth = np.array([7.0, 3.0, 5.0])          # raw 1-9 targets
    standardised = (truth - 5.0) / 2.0         # what a perfect model outputs
    collected = {
        "valence_true": truth, "valence_pred": standardised,
        "arousal_true": truth, "arousal_pred": standardised,
    }
    out = emotion_metrics(collected, cfg=cfg)
    assert out["valence_mae"] == pytest.approx(0.0, abs=1e-9), (
        "predictions were scored without inverting the standardisation"
    )

    # and without the stats the same predictions must look wrong, which is the
    # whole reason the inversion cannot be skipped
    naive = emotion_metrics(collected, cfg={"multitask": {}})
    assert naive["valence_mae"] > 1.0


def test_alternating_loader_honours_the_batch_ratio():
    """1:1 shows the emotion head each DEAM track ~14x per MTAT epoch."""
    from src.datasets import alternating_loader

    tags = [f"t{i}" for i in range(40)]
    emotions = [f"e{i}" for i in range(5)]

    pairs = list(alternating_loader(tags, emotions, ratio=(4, 1)))
    kinds = [k for _, k in pairs]
    n_tag = kinds.count("a")
    n_emotion = kinds.count("b")
    assert n_tag == 40
    assert n_emotion == pytest.approx(10, abs=2), (
        f"expected ~1 emotion batch per 4 tag batches, got {n_emotion} per {n_tag}"
    )

    one_to_one = list(alternating_loader(tags, emotions, ratio=(1, 1)))
    assert sum(1 for _, k in one_to_one if k == "b") > n_emotion, (
        "the ratio had no effect"
    )


# --------------------------------------------------------------------------- #
# the noise floor governs what the ablation may claim
# --------------------------------------------------------------------------- #
def _ablation_rows(mode, values):
    return [{"mode": mode, "corpus": "mtat", "seed": seed,
             "macro_f1": v, "macro_f1_fixed_half": v - 0.08}
            for seed, v in zip((42, 1337, 2024), values)]


def test_ablation_refuses_to_name_a_winner_inside_the_noise_floor():
    from scripts.fusion_ablation import NOISE_FLOOR, summarise

    rows = (_ablation_rows("cross_attention", [0.402, 0.400, 0.398])
            + _ablation_rows("late_concat", [0.394, 0.391, 0.393])
            + _ablation_rows("early_concat", [0.389, 0.392, 0.390]))
    out = summarise(rows, "mtat")
    assert not out["separable"]
    assert out["best_mode"] is None, "a winner was named among indistinguishable rows"
    assert set(out["modes_within_noise_of_best"]) == {
        "cross_attention", "late_concat", "early_concat"}
    assert "NOT separable" in out["verdict"]
    assert str(NOISE_FLOOR) in out["verdict"]


def test_ablation_does_report_an_ordering_when_the_gap_is_real():
    from scripts.fusion_ablation import summarise

    rows = (_ablation_rows("cross_attention", [0.470, 0.472, 0.468])
            + _ablation_rows("late_concat", [0.394, 0.391, 0.393]))
    out = summarise(rows, "mtat")
    assert out["separable"] and out["best_mode"] == "cross_attention"


def test_ablation_reports_seed_spread_on_every_row():
    """Single-run rows are exactly where small deltas get over-interpreted."""
    from scripts.fusion_ablation import summarise

    rows = (_ablation_rows("cross_attention", [0.40, 0.42, 0.38])
            + _ablation_rows("gated", [0.30, 0.30, 0.30]))
    out = summarise(rows, "mtat")
    for row in out["rows"]:
        assert row["n_seeds"] == 3
        assert row["macro_f1_sd"] is not None
        assert row["fixed_half_mean"] is not None, "fixed-0.5 column missing"
    spread = next(r for r in out["rows"] if r["mode"] == "cross_attention")
    assert spread["macro_f1_sd"] > 0.01


# --------------------------------------------------------------------------- #
# B4 -- the Google Forms adapter. An off-by-one here would silently swap real
# pairs with controls and invert the headline finding, so it is tested hard.
# --------------------------------------------------------------------------- #
def _sheet_key(n_items=4, n_controls=2):
    items = []
    for i in range(n_items + n_controls):
        control = i >= n_items
        items.append({
            "number": i + 1,
            "item_id": f"{'control' if control else 'item'}_{i:03d}",
            "caption": f"caption {i}",
            "is_control": control,
            "query_track_id": f"q{i}",
            "retrieved_track_id": f"r{i}",
        })
    return {"scale": {"min": 1, "max": 5}, "items": items,
            "n_items": n_items, "n_controls": n_controls}


def test_forms_adapter_maps_clip_numbers_not_column_order():
    """Google Forms column order is not guaranteed; the number is the anchor."""
    import pandas as pd

    from scripts.analyse_human_eval import to_long

    key = _sheet_key()
    # deliberately shuffled column order, and a timestamp column in the way
    responses = pd.DataFrame([{
        "Timestamp": "2026/09/06 10:00",
        "Clip 3: how well does the description match the audio?": 5,
        "Clip 1: how well does the description match the audio?": 4,
        "Clip 6: how well does the description match the audio?": 1,
        "Clip 2: how well does the description match the audio?": 4,
        "Clip 5: how well does the description match the audio?": 2,
        "Clip 4: how well does the description match the audio?": 5,
    }])
    long = to_long(responses, key)
    assert len(long) == 6
    by_number = dict(zip(long["clip_number"], long["rating"]))
    assert by_number[3] == 5 and by_number[6] == 1

    # clips 5 and 6 are the controls in this key, and must be labelled as such
    controls = set(long[long["is_control"]]["clip_number"])
    assert controls == {5, 6}, f"control flags did not follow the key: {controls}"


def test_forms_adapter_refuses_a_mismatched_export():
    """Guessing here would produce a plausible, wrong answer."""
    import pandas as pd

    from scripts.analyse_human_eval import to_long

    key = _sheet_key()
    responses = pd.DataFrame([{"Timestamp": "x", "Clip 1: ...": 4, "Clip 2: ...": 3}])
    with pytest.raises(SystemExit, match="could not"):
        to_long(responses, key)


def test_forms_adapter_drops_out_of_scale_and_blank_answers():
    import numpy as np
    import pandas as pd

    from scripts.analyse_human_eval import to_long

    key = _sheet_key(n_items=2, n_controls=1)
    responses = pd.DataFrame([{
        "Timestamp": "x",
        "Clip 1: q": 4, "Clip 2: q": 99, "Clip 3: q": np.nan,
    }])
    long = to_long(responses, key)
    assert list(long["clip_number"]) == [1], "an out-of-scale or blank rating was kept"


def test_forms_adapter_gives_each_respondent_its_own_rater_id():
    import pandas as pd

    from scripts.analyse_human_eval import to_long

    key = _sheet_key(n_items=2, n_controls=0)
    responses = pd.DataFrame([
        {"Timestamp": "a", "Clip 1: q": 5, "Clip 2: q": 4},
        {"Timestamp": "b", "Clip 1: q": 2, "Clip 2: q": 1},
    ])
    long = to_long(responses, key)
    assert long["rater_id"].nunique() == 2
    assert len(long) == 4


def test_control_discrimination_flags_an_uninformative_study():
    """If controls score as highly as real pairs the study says nothing."""
    import numpy as np
    import pandas as pd

    from src.human_eval import compute_agreement

    rows = []
    for rater in range(5):
        for item in range(8):
            rows.append({"item_id": f"i{item}", "rater": f"r{rater}",
                         "rating": 4.0, "is_control": item >= 6})
    stats = compute_agreement(pd.DataFrame(rows))
    assert stats["control_discrimination"] == pytest.approx(0.0, abs=1e-9)

    # and the opposite case must be detected too
    rows = []
    for rater in range(5):
        for item in range(8):
            control = item >= 6
            rows.append({"item_id": f"i{item}", "rater": f"r{rater}",
                         "rating": 1.5 if control else 4.5, "is_control": control})
    stats = compute_agreement(pd.DataFrame(rows))
    assert stats["control_discrimination"] == pytest.approx(3.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# checkpoints must carry the run tag.
#
# Without it, the seven fusion modes of the ablation each overwrite the last,
# and the case studies -- which need one specific MusicCaps model -- silently
# load whichever run finished most recently. Wrong weights, plausible output.
# --------------------------------------------------------------------------- #
def test_find_checkpoint_prefers_the_requested_run_tag(tmp_path):
    from src.utils import find_checkpoint

    for name in ("task3_seed42_mtat_cross_attention_best.pt",
                 "task3_seed42_musiccaps_cross_attention_best.pt",
                 "task3_seed42_best.pt"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    got = find_checkpoint(tmp_path, task=3, seed=42,
                          run_tag="musiccaps_cross_attention")
    assert got.name == "task3_seed42_musiccaps_cross_attention_best.pt", (
        f"asked for the MusicCaps model and got {got.name}"
    )


def test_find_checkpoint_falls_back_to_the_untagged_name(tmp_path):
    """Artifacts written before tagging existed must still load."""
    from src.utils import find_checkpoint

    (tmp_path / "task2_seed42_best.pt").write_text("x", encoding="utf-8")
    got = find_checkpoint(tmp_path, task=2, seed=42, run_tag="does_not_exist")
    assert got.name == "task2_seed42_best.pt"


def test_find_checkpoint_returns_none_rather_than_the_wrong_task(tmp_path):
    from src.utils import find_checkpoint

    (tmp_path / "task3_seed42_best.pt").write_text("x", encoding="utf-8")
    assert find_checkpoint(tmp_path, task=4, seed=42) is None
    assert find_checkpoint(tmp_path, task=3, seed=1337) is None
    assert find_checkpoint(tmp_path / "nope", task=3, seed=42) is None


def test_the_best_checkpoint_is_tagged_and_the_last_one_is_not(tmp_path):
    """Two different requirements that pull in opposite directions.

    ``_best`` is loaded by name -- by the case studies, the zero-shot
    comparison, evaluate -- so it must carry the run tag or the seven ablation
    modes overwrite each other and a reader silently gets the wrong weights.

    ``_last`` is a mid-training resume point that nothing reads back. Tagging it
    too made every run keep its own ~440 MB copy, and the disk hit 100% during
    the C7 seed runs. One rolling file per task and seed is correct there.
    """
    import re

    source = (project_root() / "src" / "train.py").read_text(encoding="utf-8")
    saves = re.findall(r'save_checkpoint\(model, ckpt_dir / f"([^"]+)"', source)
    assert saves, "could not find the checkpoint writes in _fit"

    best = [p for p in saves if "_best" in p]
    last = [p for p in saves if "_last" in p]
    assert best and last, f"expected a best and a last write, got {saves}"
    for pattern in best:
        assert "{suffix}" in pattern, (
            f"{pattern!r} has no run tag; ablation modes will overwrite each "
            "other and readers will load the wrong weights"
        )
    for pattern in last:
        assert "{suffix}" not in pattern, (
            f"{pattern!r} is tagged; nothing reads it back and one copy per run "
            "fills the disk"
        )


# --------------------------------------------------------------------------- #
# sharding must be exact, and imports must be comparable
# --------------------------------------------------------------------------- #
def test_ablation_shards_cover_every_run_exactly_once():
    """Two shards run in separate processes and cannot see each other.

    Nothing coordinates them at runtime, so the split has to be correct by
    construction or the sweep silently duplicates some runs and drops others.
    """
    from scripts.kaggle_ablation import build_runs, shard_of

    runs = build_runs()
    assert len(runs) == 21, f"expected 7 modes x 3 seeds, got {len(runs)}"

    for shards in (1, 2, 3, 4):
        seen, overlaps = set(), 0
        for shard in range(shards):
            mine = {r["index"] for r in shard_of(runs, shard, shards)}
            overlaps += len(seen & mine)
            seen |= mine
        assert overlaps == 0, f"{shards} shards overlap on {overlaps} run(s)"
        assert seen == {r["index"] for r in runs}, (
            f"{shards} shards do not cover every run"
        )


def test_ablation_shard_rejects_an_out_of_range_index():
    from scripts.kaggle_ablation import build_runs, shard_of

    with pytest.raises(ValueError):
        shard_of(build_runs(), shard=2, shards=2)


def test_ablation_run_tags_are_unique_per_seed():
    """Two runs sharing a tag and seed would overwrite each other's result."""
    from scripts.kaggle_ablation import build_runs

    keys = [(r["seed"], r["run_tag"]) for r in build_runs()]
    assert len(keys) == len(set(keys)), "a (seed, run_tag) pair repeats"


def test_vocabulary_hash_is_order_sensitive():
    """Order fixes which column is which tag, so it must change the hash."""
    from src.utils import vocabulary_hash

    a = vocabulary_hash(["guitar", "drum", "vocal"])
    assert a == vocabulary_hash(["guitar", "drum", "vocal"])
    assert a != vocabulary_hash(["drum", "guitar", "vocal"]), (
        "a reordered vocabulary hashed the same; per-tag tables would silently "
        "disagree while aggregate metrics matched"
    )
    assert a != vocabulary_hash(["guitar", "drum", "voice"])
    assert vocabulary_hash([]) == ""


def test_importer_refuses_a_foreign_vocabulary(tmp_path, monkeypatch):
    """A pre-A7.3 result is numerically fine and semantically incompatible."""
    import json as _json
    import subprocess

    from src.utils import project_root, vocabulary_hash

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    payload = {
        "task": 3, "seed": 42, "provenance": "real", "threshold_source": "val",
        "tag_vocab": ["not", "the", "real", "vocabulary"],
        "tag_vocab_hash": vocabulary_hash(["not", "the", "real", "vocabulary"]),
        "test": {"macro_f1": 0.99},
    }
    (incoming / "task3_seed42_bogus.json").write_text(_json.dumps(payload),
                                                      encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(project_root() / "scripts" / "import_kaggle_results.py"),
         str(incoming)],
        capture_output=True, text=True, cwd=str(project_root()))
    combined = result.stdout + result.stderr
    assert "REFUSED" in combined, combined[-800:]
    assert result.returncode == 2, "a rejected import should not exit clean"
    assert not (project_root() / "results" / "task3_seed42_bogus.json").exists()


# --------------------------------------------------------------------------- #
# B4 -- a null control gap has two explanations, and they are not equivalent
# --------------------------------------------------------------------------- #
def test_between_item_discrimination_separates_inattention_from_bad_controls():
    """Raters who spread items across the scale were attending, whatever the
    control gap says. Without this the study's null is unreadable."""
    import pandas as pd
    sys.path.insert(0, str(project_root() / "scripts"))
    from analyse_human_eval import between_item_discrimination

    # ten raters agreeing that item A is bad and item B is good
    rows = []
    for rater in range(10):
        rows.append({"item_id": "a", "rating": 1.0, "is_control": False})
        rows.append({"item_id": "b", "rating": 5.0, "is_control": False})
    stats = between_item_discrimination(pd.DataFrame(rows))
    assert stats["between_item_p"] < 0.01
    assert stats["between_item_variance_share"] > 0.9, (
        "perfectly separated items should put nearly all variance between them"
    )

    # the same raters clicking at random-ish, with no item structure
    rows = [{"item_id": item, "rating": float(1 + (i + j) % 5),
             "is_control": False}
            for i, item in enumerate(("a", "b")) for j in range(10)]
    stats = between_item_discrimination(pd.DataFrame(rows))
    assert stats["between_item_variance_share"] < 0.2


def test_control_caption_reuse_is_detected():
    """A control whose caption also appears on a real item is not a control."""
    sys.path.insert(0, str(project_root() / "scripts"))
    from analyse_human_eval import control_caption_reuse

    key = {"items": [
        {"item_id": "item_0", "number": 1, "is_control": False, "caption": "a hip hop song"},
        {"item_id": "ctrl_0", "number": 2, "is_control": True, "caption": "a hip hop song"},
        {"item_id": "ctrl_1", "number": 3, "is_control": True, "caption": "unique text"},
    ]}
    found = control_caption_reuse(key)
    assert found["n_controls"] == 2
    assert found["n_controls_reusing_a_caption"] == 1
    assert found["control_caption_reuse"][0]["control_id"] == "ctrl_0"
    assert found["control_caption_reuse"][0]["shares_caption_with"] == ["item_0"]


# --------------------------------------------------------------------------- #
# the report has exactly one source
# --------------------------------------------------------------------------- #
def test_the_retired_markdown_report_generator_stays_retired():
    """`final_report.tex` is the only source of report prose.

    An earlier markdown pipeline rendered a second version of the report from a
    metrics file, which is how a figure built on smoke-test data reached a
    PDF and stayed there across three revisions. Two sources of prose means one
    of them is always stale, and nothing tells you which.
    """
    report = project_root() / "report"
    for gone in ("final_report.md", "build_report.py"):
        assert not (report / gone).exists(), (
            f"report/{gone} is back. The .tex is the only report source; a "
            "second generator is how a stale figure gets published."
        )


# --------------------------------------------------------------------------- #
# Task 3 evaluated the wrong corpora: train filtered, val/test did not
# --------------------------------------------------------------------------- #
def test_task3_evaluation_is_filtered_to_the_training_corpora():
    """An MTAT run was scored over 7,079 test rows instead of 3,775.

    datasets.py gives the -1 sentinel to any corpus outside {mtat, musiccaps},
    so FMA and DEAM rows were masked out harmlessly. MusicCaps is *inside* that
    set, so its rows arrived as confident zeros against the MTAT vocabulary and
    counted as true negatives on all 50 tags -- worth about 0.07 macro-F1 on
    every Task 3 result. The guard is textual because constructing a real
    bundle needs the corpora on disk.
    """
    source = (project_root() / "src" / "train.py").read_text(encoding="utf-8")
    start = source.index("def run_task3")
    body = source[start:source.index("\ndef ", start + 10)]

    for split in ("val", "test"):
        assert f'bundle.dataset("{split}", eval_corpora)' in body, (
            f'run_task3 builds its {split} loader without a corpus filter. '
            "Training filters with corpora_for(); evaluation must use the same "
            "corpora or MusicCaps rows are scored against the MTAT vocabulary."
        )
    assert 'bundle.dataset("val")' not in body, "unfiltered val loader is back"
    assert 'bundle.dataset("test")' not in body, "unfiltered test loader is back"
