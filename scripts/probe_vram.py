#!/usr/bin/env python
"""measure real peak VRAM for every configuration we might train locally.

The Day 1-2 figures (302 MB / 297 MB) were measured with a 2-layer, 64-dim test
encoder, not a real BERT, and are worthless for capacity planning. This runs a
genuine forward + backward for each config on the actual GPU and reports
`torch.cuda.max_memory_allocated()`, so the local-vs-Kaggle routing decision is
made on evidence.

    python scripts/probe_vram.py [--budget-mb 4096] [--quick]

Writes `state/vram_probe.json`; the table lives in `state/vram_report.md`.

Each config runs in a subprocess-free but freshly-reset allocator state
(`empty_cache` + `reset_peak_memory_stats`), and an OOM is caught and recorded
as a data point rather than crashing the sweep -- "does not fit" is exactly what
we are trying to learn.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.utils import (  # noqa: E402
    autocast_ctx,
    count_parameters,
    get_logger,
    load_config,
    make_grad_scaler,
    project_root,
    set_seed,
)

LOGGER = get_logger("gbmc.probe_vram")

#: 4 GB card, but Windows WDDM reserves a slice and fragmentation costs more.
#: Anything above this fraction of the budget is "tight" rather than "fits".
SAFE_FRACTION = 0.80


def _fake_batch(batch_size: int, n_nodes: int, cfg, device):
    """A PyG batch of realistically sized graphs, on device."""
    from torch_geometric.data import Batch

    from src.graph_builder import build_segment_graph

    rng = np.random.default_rng(0)
    graphs = []
    for _ in range(batch_size):
        feats = rng.normal(size=(n_nodes, int(cfg["graph"]["node_feat_dim"]))).astype(np.float32)
        graphs.append(build_segment_graph(feats, cfg, text="a caption", dataset="probe"))
    return Batch.from_data_list(graphs).to(device)


def _fake_text(batch_size: int, seq_len: int, device, vocab: int = 30522):
    ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    return ids, torch.ones_like(ids)


def _measure(fn, device) -> dict:
    """Run one forward+backward and report the peak allocation it needed."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        params = fn()
        peak = (torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else 0.0
        # reserved, not allocated, is what actually collides with the 4 GB wall:
        # the caching allocator holds whole segments, so fragmentation shows up
        # here and nowhere else.
        reserved = (torch.cuda.max_memory_reserved() / 1024**2) if device.type == "cuda" else 0.0
        return {"ok": True, "peak_mb": round(peak, 1),
                "reserved_mb": round(reserved, 1), "params": params, "error": None}
    except torch.cuda.OutOfMemoryError:
        return {"ok": False, "peak_mb": None, "reserved_mb": None,
                "params": None, "error": "CUDA OOM"}
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            return {"ok": False, "peak_mb": None, "reserved_mb": None,
                    "params": None, "error": "CUDA OOM"}
        return {"ok": False, "peak_mb": None, "reserved_mb": None, "params": None,
                "error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # pragma: no cover - unexpected
        return {"ok": False, "peak_mb": None, "params": None,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"}
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _maybe_grad_checkpoint(module, enable: bool) -> bool:
    """Turn on HuggingFace gradient checkpointing if the model supports it.

    The caller must also have put the model in train mode: HF skips
    checkpointing entirely when ``not self.training``, and ``from_pretrained``
    hands back an eval-mode model. Missing that is how this bug announced
    itself -- the ckpt rows came out byte-identical to the non-ckpt ones.
    """
    if not enable:
        return False
    target = getattr(module, "bert", module)
    if not hasattr(target, "gradient_checkpointing_enable"):
        return False
    target.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(target, "config"):
        target.config.use_cache = False
    return bool(getattr(target, "is_gradient_checkpointing", True))


# --------------------------------------------------------------------------- #
# per-task probes
# --------------------------------------------------------------------------- #
def probe_task1(cfg, device, model_name, batch, seq, amp, grad_ckpt):
    from src.bert_encoder import BertTagClassifier

    def run():
        set_seed(0)
        model = BertTagClassifier(n_tags=50, model_name=model_name,
                                  freeze_mode="full_ft").to(device)
        model.train()                      # required for checkpointing to engage
        used = _maybe_grad_checkpoint(model.encoder, grad_ckpt)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scaler = make_grad_scaler(amp, device.type)
        ids, mask = _fake_text(batch, seq, device)
        target = torch.randint(0, 2, (batch, 50), device=device).float()
        with autocast_ctx(amp, device.type):
            logits = model(ids, mask)
            loss = F.binary_cross_entropy_with_logits(logits.float(), target)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        return {"trainable": count_parameters(model), "grad_ckpt_applied": used}
    return run


def probe_task2(cfg, device, batch, amp, n_nodes=32):
    from src.gnn_model import GNNClassifier

    def run():
        set_seed(0)
        model = GNNClassifier(
            n_tags=50, in_dim=int(cfg["graph"]["node_feat_dim"]),
            hidden_dim=int(cfg["gnn"]["hidden_dim"]),
            num_layers=int(cfg["gnn"]["num_layers"]), conv=str(cfg["gnn"]["conv"]),
            readout=str(cfg["gnn"]["readout"]),
        ).to(device)
        model.train()
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = make_grad_scaler(amp, device.type)
        data = _fake_batch(batch, n_nodes, cfg, device)
        target = torch.randint(0, 2, (batch, 50), device=device).float()
        with autocast_ctx(amp, device.type):
            loss = F.binary_cross_entropy_with_logits(
                model(data)["tag_logits"].float(), target)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        return {"trainable": count_parameters(model), "grad_ckpt_applied": False}
    return run


def probe_task3(cfg, device, model_name, batch, seq, amp, grad_ckpt, n_nodes=32):
    from src.bert_encoder import BertTextEncoder
    from src.fusion_model import GNNBertFusion
    from src.gnn_model import GNNEncoder

    def run():
        set_seed(0)
        gnn = GNNEncoder(in_dim=int(cfg["graph"]["node_feat_dim"]),
                         hidden_dim=int(cfg["gnn"]["hidden_dim"]),
                         num_layers=int(cfg["gnn"]["num_layers"]),
                         conv=str(cfg["gnn"]["conv"]), readout=str(cfg["gnn"]["readout"]))
        bert = BertTextEncoder(model_name, freeze_mode="full_ft")
        model = GNNBertFusion(gnn, bert, mode=str(cfg["fusion"]["mode"]),
                              shared_dim=int(cfg["fusion"]["shared_dim"]),
                              n_heads=int(cfg["fusion"]["n_heads"]), n_tags=50).to(device)
        model.train()                      # required for checkpointing to engage
        used = _maybe_grad_checkpoint(bert, grad_ckpt)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scaler = make_grad_scaler(amp, device.type)
        data = _fake_batch(batch, n_nodes, cfg, device)
        ids, mask = _fake_text(batch, seq, device)
        target = torch.randint(0, 2, (batch, 50), device=device).float()
        with autocast_ctx(amp, device.type):
            out = model(data, ids, mask)
            loss = F.binary_cross_entropy_with_logits(out["tag_logits"].float(), target)
            loss = loss + 0.1 * (out["valence"].float().mean() ** 2)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        return {"trainable": count_parameters(model), "grad_ckpt_applied": used}
    return run


def probe_task4(cfg, device, model_name, batch, amp, n_nodes=32, precomputed=True):
    from src.bert_encoder import BertTextEncoder
    from src.contrastive import DualEncoder, symmetric_info_nce
    from src.gnn_model import GNNEncoder

    def run():
        set_seed(0)
        gnn = GNNEncoder(in_dim=int(cfg["graph"]["node_feat_dim"]),
                         hidden_dim=int(cfg["gnn"]["hidden_dim"]),
                         num_layers=int(cfg["gnn"]["num_layers"]),
                         conv=str(cfg["gnn"]["conv"]), readout=str(cfg["gnn"]["readout"]))
        bert = BertTextEncoder(model_name, freeze_mode="frozen_probe")
        model = DualEncoder(gnn, bert, shared_dim=int(cfg["fusion"]["shared_dim"]),
                            temperature_init=0.07).to(device)
        model.train()
        if precomputed:
            model.set_precomputed_text(torch.randn(batch * 2, bert.hidden_size))
        optimiser = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3)
        scaler = make_grad_scaler(amp, device.type)
        data = _fake_batch(batch, n_nodes, cfg, device)
        with autocast_ctx(amp, device.type):
            g = model.encode_graph(data)
            t = (model.encode_text_precomputed(list(range(batch))) if precomputed
                 else model.encode_text(*_fake_text(batch, 128, device)))
            loss = symmetric_info_nce(g, t, model.get_logit_scale())
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        return {"trainable": count_parameters(model), "grad_ckpt_applied": False}
    return run


# --------------------------------------------------------------------------- #
def build_matrix(quick: bool) -> list[dict]:
    """The sweep from the Phase A spec, plus the configs we might actually run."""
    dbert, bbase = "distilbert-base-uncased", "bert-base-uncased"
    rows = [
        {"task": "Task 1 distilbert", "model": dbert, "batch": 8, "seq": 128, "grad_ckpt": False},
        {"task": "Task 1 distilbert", "model": dbert, "batch": 16, "seq": 128, "grad_ckpt": False},
        {"task": "Task 1 distilbert", "model": dbert, "batch": 32, "seq": 128, "grad_ckpt": False},
        {"task": "Task 1 distilbert", "model": dbert, "batch": 32, "seq": 128, "grad_ckpt": True},
        {"task": "Task 1 bert-base", "model": bbase, "batch": 8, "seq": 128, "grad_ckpt": False},
        {"task": "Task 1 bert-base", "model": bbase, "batch": 16, "seq": 128, "grad_ckpt": False},
        {"task": "Task 1 bert-base", "model": bbase, "batch": 16, "seq": 128, "grad_ckpt": True},
        {"task": "Task 2 GNN", "model": None, "batch": 8, "seq": None, "grad_ckpt": False},
        {"task": "Task 2 GNN", "model": None, "batch": 32, "seq": None, "grad_ckpt": False},
        {"task": "Task 2 GNN", "model": None, "batch": 128, "seq": None, "grad_ckpt": False},
        {"task": "Task 3 fusion distilbert", "model": dbert, "batch": 8, "seq": 128, "grad_ckpt": False},
        {"task": "Task 3 fusion distilbert", "model": dbert, "batch": 8, "seq": 128, "grad_ckpt": True},
        {"task": "Task 3 fusion distilbert", "model": dbert, "batch": 16, "seq": 128, "grad_ckpt": False},
        {"task": "Task 3 fusion bert-base", "model": bbase, "batch": 8, "seq": 128, "grad_ckpt": False},
        {"task": "Task 3 fusion bert-base", "model": bbase, "batch": 8, "seq": 128, "grad_ckpt": True},
        {"task": "Task 4 frozen + precomputed", "model": dbert, "batch": 256, "seq": None, "grad_ckpt": False},
        {"task": "Task 4 frozen + precomputed", "model": dbert, "batch": 512, "seq": None, "grad_ckpt": False},
        # ---- push until it breaks, so the report can name the actual ceiling ----
        {"task": "Task 1 bert-base", "model": bbase, "batch": 32, "seq": 128, "grad_ckpt": False},
        {"task": "Task 1 bert-base", "model": bbase, "batch": 32, "seq": 128, "grad_ckpt": True},
        {"task": "Task 1 bert-base", "model": bbase, "batch": 64, "seq": 128, "grad_ckpt": False},
        {"task": "Task 1 bert-base", "model": bbase, "batch": 64, "seq": 128, "grad_ckpt": True},
        {"task": "Task 1 bert-base seq256", "model": bbase, "batch": 32, "seq": 256, "grad_ckpt": False},
        {"task": "Task 1 bert-base seq256", "model": bbase, "batch": 32, "seq": 256, "grad_ckpt": True},
        {"task": "Task 3 fusion bert-base", "model": bbase, "batch": 16, "seq": 128, "grad_ckpt": False},
        {"task": "Task 3 fusion bert-base", "model": bbase, "batch": 32, "seq": 128, "grad_ckpt": False},
        {"task": "Task 3 fusion bert-base", "model": bbase, "batch": 32, "seq": 128, "grad_ckpt": True},
        {"task": "Task 3 fusion bert-base", "model": bbase, "batch": 64, "seq": 128, "grad_ckpt": True},
        {"task": "Task 4 frozen + precomputed", "model": dbert, "batch": 1024, "seq": None, "grad_ckpt": False},
        {"task": "Task 4 frozen + precomputed", "model": dbert, "batch": 2048, "seq": None, "grad_ckpt": False},
    ]
    if quick:
        rows = [r for r in rows if r["batch"] <= 32]
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Measure real peak VRAM per config.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--budget-mb", type=float, default=4096.0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--amp", default="on", choices=["on", "off", "both"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        LOGGER.error("no CUDA device; a VRAM probe on CPU measures nothing")
        return 1

    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
    LOGGER.info("probing on %s (%.0f MB total, %.0f MB safe budget)",
                torch.cuda.get_device_name(0), total_mb, args.budget_mb * SAFE_FRACTION)

    amp_modes = {"on": [True], "off": [False], "both": [True, False]}[args.amp]
    results = []
    for row in build_matrix(args.quick):
        for amp in amp_modes:
            task = row["task"]
            if task.startswith("Task 1"):
                fn = probe_task1(cfg, device, row["model"], row["batch"], row["seq"],
                                 amp, row["grad_ckpt"])
            elif task.startswith("Task 2"):
                fn = probe_task2(cfg, device, row["batch"], amp)
            elif task.startswith("Task 3"):
                fn = probe_task3(cfg, device, row["model"], row["batch"], row["seq"],
                                 amp, row["grad_ckpt"])
            else:
                fn = probe_task4(cfg, device, row["model"], row["batch"], amp)

            measured = _measure(fn, device)
            record = {**row, "amp": amp, **measured}
            if measured["ok"]:
                # judge against reserved: that is what the driver actually holds
                peak = max(measured["peak_mb"], measured.get("reserved_mb") or 0.0)
                record["fits"] = peak < args.budget_mb * SAFE_FRACTION
                record["verdict"] = ("local" if record["fits"]
                                     else "tight" if peak < args.budget_mb else "kaggle")
            else:
                record["fits"] = False
                record["verdict"] = "kaggle"
            results.append(record)
            LOGGER.info("%-30s b=%-4d amp=%-5s ckpt=%-5s -> %s",
                        task, row["batch"], amp, row["grad_ckpt"],
                        f"{measured['peak_mb']} MB" if measured["ok"] else measured["error"])

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "total_mb": round(total_mb, 1),
        "budget_mb": args.budget_mb,
        "safe_fraction": SAFE_FRACTION,
        "torch": torch.__version__,
        "note": ("sm_75 has no tensor cores and no bf16, so AMP here buys memory, "
                 "not speed. fp16 is the only useful autocast dtype."),
        "results": results,
    }
    out = Path(args.out) if args.out else project_root() / "state" / "vram_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'config':<30} {'batch':>6} {'seq':>5} {'amp':>5} {'ckpt':>5} "
          f"{'alloc MB':>9} {'resvd MB':>9}  verdict")
    print("-" * 92)
    for r in results:
        alloc = f"{r['peak_mb']:.0f}" if r["ok"] else "OOM"
        resvd = f"{r['reserved_mb']:.0f}" if r["ok"] and r.get("reserved_mb") else "-"
        print(f"{r['task']:<30} {r['batch']:>6} {str(r['seq'] or '-'):>5} {str(r['amp']):>5} "
              f"{str(r['grad_ckpt']):>5} {alloc:>9} {resvd:>9}  {r['verdict']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
