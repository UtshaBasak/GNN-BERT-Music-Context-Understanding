#!/usr/bin/env python
"""check that the BERT checkpoints we actually use load and work.

Run this before spending Kaggle hours. It verifies, per model, that the
tokenizer loads and is a *fast* one, that the body loads, and that attention
weights actually come back — the last of which is easy to lose silently, because
`transformers` defaults to the SDPA kernel and SDPA returns `None` for
`output_attentions` without raising.

    python scripts/probe_env.py [--models distilbert-base-uncased bert-base-uncased]

Writes `state/bert_probe.json`; the human-readable summary lives in
`state/env_report.md`.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.utils import get_logger, project_root  # noqa: E402

LOGGER = get_logger("gbmc.probe_env")

DEFAULT_MODELS = ["distilbert-base-uncased", "bert-base-uncased"]


def probe_model(name: str) -> dict:
    """Load one checkpoint end to end and report exactly what worked."""
    from transformers import AutoModel, AutoTokenizer

    row: dict = {
        "model": name, "tokenizer_ok": False, "fast_tokenizer": None,
        "tokenizer_class": None, "model_ok": False, "attentions_ok": False,
        "n_attention_layers": 0, "attention_shape": None, "params": None,
        "hidden_size": None, "n_layers": None, "load_seconds": None, "error": None,
    }
    started = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        row["tokenizer_ok"] = True
        row["fast_tokenizer"] = bool(getattr(tokenizer, "is_fast", False))
        row["tokenizer_class"] = type(tokenizer).__name__

        # eager, not SDPA -- see src.bert_encoder.ATTENTION_NOTE
        model = AutoModel.from_pretrained(name, attn_implementation="eager")
        row["model_ok"] = True
        row["params"] = int(sum(p.numel() for p in model.parameters()))
        row["hidden_size"] = int(model.config.hidden_size)
        row["n_layers"] = int(getattr(model.config, "num_hidden_layers",
                                      getattr(model.config, "n_layers", -1)))

        encoded = tokenizer("test caption", return_tensors="pt")
        with torch.no_grad():
            out = model(**encoded, output_attentions=True)
        if not out.attentions:
            raise RuntimeError(
                "output_attentions returned nothing -- the backbone is not on the "
                "eager attention path, and attention_viz would silently draw nothing"
            )
        row["attentions_ok"] = True
        row["n_attention_layers"] = len(out.attentions)
        row["attention_shape"] = list(out.attentions[-1].shape)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        LOGGER.error("%s FAILED: %s", name, row["error"])
    row["load_seconds"] = round(time.time() - started, 1)
    return row


def probe_wrapper(model_name: str) -> dict:
    """Confirm the project's own encoder honours the eager path."""
    out: dict = {"model": model_name, "attentions_ok": None, "error": None}
    try:
        from src.bert_encoder import BertTextEncoder, load_tokenizer

        encoder = BertTextEncoder(model_name, freeze_mode="frozen_probe",
                                  output_attentions=True)
        tokenizer = load_tokenizer(model_name)
        batch = tokenizer(["a mellow piano ballad"], return_tensors="pt")
        with torch.no_grad():
            cls, _tokens, attentions = encoder(
                batch["input_ids"], batch["attention_mask"], return_attentions=True
            )
        out["attentions_ok"] = bool(attentions) and len(attentions) > 0
        out["n_attention_layers"] = len(attentions or [])
        out["cls_shape"] = list(cls.shape)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Probe BERT checkpoint loadability.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    import transformers

    rows = [probe_model(name) for name in args.models]
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models": rows,
        "wrapper": probe_wrapper(args.models[0]) if args.models else {},
    }

    out = Path(args.out) if args.out else project_root() / "state" / "bert_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{'model':<28} {'tok':<5} {'fast':<5} {'body':<5} {'attn':<5} {'layers':<7} params")
    print("-" * 78)
    for row in rows:
        print(f"{row['model']:<28} "
              f"{'OK' if row['tokenizer_ok'] else 'FAIL':<5} "
              f"{str(row['fast_tokenizer']):<5} "
              f"{'OK' if row['model_ok'] else 'FAIL':<5} "
              f"{'OK' if row['attentions_ok'] else 'FAIL':<5} "
              f"{row['n_attention_layers']:<7} "
              f"{row['params'] if row['params'] else '-'}")
        if row["error"]:
            print(f"    error: {row['error']}")
    print(f"\nwrote {out}")

    failed = [r["model"] for r in rows if not (r["model_ok"] and r["attentions_ok"])]
    if failed:
        LOGGER.error("these models are unusable as configured: %s. Pin a working "
                     "transformers version and note it in requirements.txt.", failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
