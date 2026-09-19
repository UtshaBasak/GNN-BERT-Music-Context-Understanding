#!/usr/bin/env python
"""zero-shot tagging through the contrastive encoder, and what it costs.

    python scripts/zero_shot_eval.py --seed 42

Task 4's dual encoder can tag without ever being trained to tag: embed each tag
name as text, embed the graph, and rank. The interesting questions are how much
that costs against supervision, and how much of the number is an artefact of
phrasing.

**Prompt sensitivity is measured, not assumed away.** Tag names are embedded
through at least three templates (`"{tag}"`, `"a recording of {tag} music"`,
`"this music sounds {tag}"`). Each template is scored separately and the spread
is reported beside the mean. A single-template zero-shot number invites a reader
to believe a precision the method does not have.

**The supervised comparison is like-for-like or it is nothing.** The reference
is the Task 3 MusicCaps run: same corpus, same 50-aspect vocabulary, same test
split. Comparing zero-shot on MusicCaps against a model trained on MTAT's
vocabulary would be meaningless, so this refuses to do it.

Thresholds obey the same rule as everywhere else: tuned on validation, frozen,
then applied once to test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics as M  # noqa: E402
from src.utils import (autocast_ctx, find_checkpoint, get_device,  # noqa: E402
                       get_logger, load_config, parse_overrides, project_root,
                       save_json, set_seed)

LOGGER = get_logger("gbmc.zeroshot")

TEMPLATES = [
    "{tag}",
    "a recording of {tag} music",
    "this music sounds {tag}",
    "an audio clip that is {tag}",
]


def _build_dual(cfg, device):
    from src.bert_encoder import BertTextEncoder
    from src.contrastive import DualEncoder
    from src.gnn_model import GNNEncoder

    gnn = GNNEncoder(in_dim=int(cfg["graph"]["node_feat_dim"]),
                     hidden_dim=int(cfg["gnn"]["hidden_dim"]),
                     num_layers=int(cfg["gnn"]["num_layers"]),
                     conv=str(cfg["gnn"]["conv"]), dropout=0.0,
                     readout=str(cfg["gnn"]["readout"]))
    bert = BertTextEncoder(cfg["bert"]["model_name"], freeze_mode="frozen_probe")
    return DualEncoder(gnn, bert, shared_dim=int(cfg["fusion"]["shared_dim"]),
                       temperature_init=float(cfg["contrastive"]["temperature_init"])
                       ).to(device)


@torch.no_grad()
def _embed_graphs(model, loader, device, amp):
    graphs, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        with autocast_ctx(amp, device.type):
            graphs.append(model.encode_graph(batch).float().cpu())
        y = getattr(batch, "y_tags", None)
        targets.append(y.float().cpu().reshape(graphs[-1].shape[0], -1))
    if not graphs:
        return torch.zeros((0, 1)), np.zeros((0, 0))
    return torch.cat(graphs), torch.cat(targets).numpy()


@torch.no_grad()
def _score(model, tags, template, tokenizer, device, max_length, graphs):
    import torch.nn.functional as F

    prompts = [template.format(tag=tag) for tag in tags]
    encoded = tokenizer(prompts, padding=True, truncation=True,
                        max_length=int(max_length), return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    text = model.encode_text(encoded["input_ids"], encoded["attention_mask"])
    g = F.normalize(graphs.to(device).float(), dim=-1)
    # cosine in [-1, 1] -> [0, 1] so the same threshold tuner applies unchanged
    return ((g @ text.t()).cpu().numpy() + 1.0) / 2.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-tag", default=None,
                        help="pick a specific Task 4 checkpoint by run tag")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args(argv)

    from src.bert_encoder import load_tokenizer
    from src.datasets import make_loader
    from src.train import DataBundle, corpora_for

    overrides = parse_overrides(args.override)

    # Zero-shot is evaluated on MusicCaps against the MusicCaps vocabulary,
    # because the supervised reference below is the Task 3 MusicCaps run. Any
    # other corpus or label space would make the comparison meaningless, so
    # these are pinned rather than merely defaulted -- and an attempt to
    # override them is refused out loud rather than quietly discarded.
    pinned = {
        "data.tag_corpora": ["musiccaps"],
        "data.caption_corpora": ["musiccaps"],
        "data.text_source": "caption_masked",
        "tags.vocab_source": "musiccaps",
    }
    clashes = [k for k in pinned if k in overrides and overrides[k] != pinned[k]]
    if clashes:
        raise SystemExit(
            "zero-shot must run on MusicCaps against the MusicCaps vocabulary "
            "so it is comparable with the Task 3 secondary run; refusing to "
            f"override {', '.join(sorted(clashes))}"
        )
    overrides.update(pinned)

    cfg = load_config(args.config, overrides)
    cfg["device"] = args.device
    set_seed(args.seed)
    device = get_device(str(cfg.get("device", "cuda")))
    amp = bool(cfg.get("amp", True))

    bundle = DataBundle(cfg, synthetic=False)
    tags = list(bundle.tag_vocab)
    LOGGER.info("zero-shot over %d MusicCaps aspects, %d templates",
                len(tags), len(TEMPLATES))

    ckpt = find_checkpoint(
        project_root() / cfg["paths"].get("checkpoints", "results/checkpoints"),
        task=4, seed=args.seed, run_tag=args.run_tag)
    if ckpt is None:
        raise SystemExit(
            f"no Task 4 checkpoint for seed {args.seed} -- train it first "
            f"(python -m src.train --task 4 --seed {args.seed})"
        )
    LOGGER.info("using checkpoint %s", ckpt.name)
    payload = torch.load(ckpt, map_location=device, weights_only=False)

    # The encoder must be the architecture the checkpoint was trained with, not
    # whatever config.yaml currently defaults to. Task 4 trains bert-base while
    # the config default is distilbert, and load_state_dict(strict=False) is
    # happy to leave 96 tensors of a 12-layer tower randomly initialised and
    # return a number that looks entirely plausible.
    trained_with = ((payload.get("config") or {}).get("bert") or {}).get("model_name")
    if trained_with and trained_with != cfg["bert"]["model_name"]:
        LOGGER.info("checkpoint was trained with %s; config says %s -- using the "
                    "checkpoint's", trained_with, cfg["bert"]["model_name"])
        cfg["bert"]["model_name"] = trained_with

    model = _build_dual(cfg, device)
    state = payload.get("model_state", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise SystemExit(
            f"{len(missing)} parameter(s) did not load from {ckpt.name} "
            f"(e.g. {missing[:3]}). The zero-shot score would be measured on a "
            "partly random encoder, so this refuses rather than reporting it."
        )
    model.eval()

    tokenizer = load_tokenizer(cfg["bert"]["model_name"])
    max_length = int(cfg["bert"]["max_length"])

    embeddings, truths = {}, {}
    for split in ("val", "test"):
        loader = make_loader(bundle.dataset(split, corpora_for(cfg, "tag")), cfg,
                             shuffle=False, seed=args.seed, num_workers=0,
                             batch_size=32)
        embeddings[split], truths[split] = _embed_graphs(model, loader, device, amp)
        LOGGER.info("%s: %d clips", split, embeddings[split].shape[0])

    per_template = []
    stacked_val, stacked_test = [], []
    for template in TEMPLATES:
        val_scores = _score(model, tags, template, tokenizer, device, max_length,
                            embeddings["val"])
        test_scores = _score(model, tags, template, tokenizer, device, max_length,
                             embeddings["test"])
        stacked_val.append(val_scores)
        stacked_test.append(test_scores)
        thresholds = M.tune_thresholds(truths["val"], val_scores)   # VAL only
        per_template.append({
            "template": template,
            "macro_f1": M.macro_f1(truths["test"], test_scores, thresholds),
            "micro_f1": M.micro_f1(truths["test"], test_scores, thresholds),
            "mean_auc_pr": M.mean_auc_pr(truths["test"], test_scores),
            "macro_f1_fixed_half": M.macro_f1(truths["test"], test_scores, 0.5),
        })
        LOGGER.info("template %-30s macro-F1 %.4f", f"{template!r}",
                    per_template[-1]["macro_f1"])

    # the ensemble: average the scores across templates, then tune once
    mean_val = np.mean(stacked_val, axis=0)
    mean_test = np.mean(stacked_test, axis=0)
    ensemble_thresholds = M.tune_thresholds(truths["val"], mean_val)
    ensemble = {
        "macro_f1": M.macro_f1(truths["test"], mean_test, ensemble_thresholds),
        "micro_f1": M.micro_f1(truths["test"], mean_test, ensemble_thresholds),
        "mean_auc_pr": M.mean_auc_pr(truths["test"], mean_test),
        "macro_f1_fixed_half": M.macro_f1(truths["test"], mean_test, 0.5),
    }

    scores = [row["macro_f1"] for row in per_template]
    spread = float(np.max(scores) - np.min(scores))

    # ---- the like-for-like supervised reference --------------------------- #
    supervised = None
    candidates = sorted((project_root() / "results").glob(
        f"task3_seed{args.seed}_musiccaps_*.json"))
    for path in candidates:
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("tag_vocab") and list(run["tag_vocab"]) != tags:
            LOGGER.warning("%s used a different vocabulary; not comparable",
                           path.name)
            continue
        test = run.get("test", {})
        if test.get("macro_f1") is None:
            continue
        if supervised is None or test["macro_f1"] > supervised["macro_f1"]:
            supervised = {"run": path.stem, "macro_f1": test["macro_f1"],
                          "micro_f1": test.get("micro_f1"),
                          "fusion_mode": run.get("fusion_mode")}

    payload = {
        "seed": args.seed,
        "corpus": "musiccaps",
        "vocab_source": "musiccaps",
        "n_tags": len(tags),
        "n_test_clips": int(embeddings["test"].shape[0]),
        "templates": TEMPLATES,
        "per_template": per_template,
        "ensemble": ensemble,
        "template_spread_macro_f1": spread,
        "threshold_source": "val",
        "supervised_reference": supervised,
        "noise_floor": 0.0288,
    }
    if supervised:
        payload["zero_shot_gap"] = supervised["macro_f1"] - ensemble["macro_f1"]

    out = save_json(payload, project_root() / "results" / f"zero_shot_seed{args.seed}.json")

    print(f"\nzero-shot tagging, MusicCaps, {len(tags)} aspects, "
          f"{payload['n_test_clips']} test clips")
    print(f"{'template':<34}{'macro-F1':>10}{'@0.5':>9}{'AUC-PR':>9}")
    print("-" * 62)
    for row in per_template:
        print(f"{row['template']!r:<34}{row['macro_f1']:>10.4f}"
              f"{row['macro_f1_fixed_half']:>9.4f}{row['mean_auc_pr']:>9.4f}")
    print(f"{'ensemble (mean of templates)':<34}{ensemble['macro_f1']:>10.4f}"
          f"{ensemble['macro_f1_fixed_half']:>9.4f}{ensemble['mean_auc_pr']:>9.4f}")
    verdict = ("ABOVE the 0.0288 noise floor: phrasing moves this result more "
               "than most modelling choices do, and a single-template number "
               "would be reporting an artefact"
               if spread > 0.0288 else
               "below the 0.0288 noise floor: phrasing is not the dominant "
               "effect here")
    print(f"\ntemplate spread: {spread:.4f} macro-F1 -- {verdict}")
    if supervised:
        print(f"supervised reference ({supervised['run']}): "
              f"{supervised['macro_f1']:.4f}")
        print(f"zero-shot gives up {payload['zero_shot_gap']:+.4f} macro-F1 "
              "against supervision on the same corpus, vocabulary and split")
    else:
        print("no like-for-like Task 3 MusicCaps run found yet; "
              "the supervised comparison is pending")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
