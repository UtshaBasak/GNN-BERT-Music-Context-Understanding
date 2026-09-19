# =============================================================================
# GNN-BERT Music Context Understanding
#
# `make smoke` is the one that matters: it runs the acceptance path end to end
# on synthetic data, on CPU, in a couple of minutes.
# =============================================================================
PYTHON ?= python
CONFIG ?= config.yaml
DEVICE ?= cuda
SEED   ?= 42
EPOCHS ?=

OVERRIDE := $(if $(EPOCHS),--override train.epochs=$(EPOCHS),)
TRAIN    := $(PYTHON) -m src.train --config $(CONFIG) --device $(DEVICE) --seed $(SEED)

.PHONY: report mel-cache vocab thresholds kaggle-payload-ablation help setup verify-data splits features graphs smoke human-eval human-eval-sheet \
        task1 task2 task3 task4 all-tasks baselines evaluate test lint clean clean-results

help:
	@echo "Targets:"
	@echo "  setup         install dependencies in the correct order (torch -> PyG -> rest)"
	@echo "  verify-data   check every dataset path, count files, report MusicCaps survival"
	@echo "  splits        build data/splits/*_manifest.csv from the raw corpora"
	@echo "  features      extract the 96-dim segment cache into data/processed/features.h5"
	@echo "  graphs        export >= 20 sample graphs as runnable examples"
	@echo "  smoke         acceptance criteria 4-8 on synthetic data, CPU"
	@echo "  task1..task4  train one task            (DEVICE=cuda SEED=42 EPOCHS=10)"
	@echo "  all-tasks     train all four, all seeds from config.eval.seeds"
	@echo "  baselines     B1 random/majority, B2 CNN, B4 PCA+MLP"
	@echo "  evaluate      regenerate every table and plot into results/"
	@echo "  graph-sanity  do similarity edges reach repeated sections?"
	@echo "  probe-env     check BERT checkpoints load with working attentions"
	@echo "  probe-vram    measure real peak VRAM per config on this GPU"
	@echo "  kaggle-payload build the upload archive (graphs + text, never mel caches)"
	@echo "  kaggle-payload-ablation  archive: feature caches + splits + code"
	@echo "  mel-cache     full-resolution mel cache for B2 (~5.8 GB, ~45 min)"
	@echo "  vocab         re-derive both tag vocabularies, train split only"
	@echo "  thresholds    bootstrap the val-tuned thresholds (100 replicates)"
	@echo "  report        fill and structurally check report/final_report.tex"
	@echo "  test          pytest"
	@echo "  clean         remove caches, checkpoints and generated results"

# ---- environment ------------------------------------------------------------
setup:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cu126
	$(PYTHON) -m pip install torch-geometric
	$(PYTHON) -m pip install -r requirements.txt
	@echo "If you need pyg-lib/torch-scatter/torch-sparse, install them from the"
	@echo "wheel index that matches your exact torch+CUDA string -- see README."

# ---- data -------------------------------------------------------------------
verify-data:
	$(PYTHON) scripts/verify_datasets.py --config $(CONFIG)

splits:
	$(PYTHON) -m src.splits --config $(CONFIG)

features:
	$(PYTHON) -m src.audio_features --config $(CONFIG)

graphs:
	$(PYTHON) scripts/build_graphs.py --config $(CONFIG) --kinds segment chord hetero
	$(PYTHON) scripts/export_sample_graphs.py --config $(CONFIG) --n 20

graph-sanity:
	$(PYTHON) scripts/graph_sanity.py --config $(CONFIG) --dataset mtat --n 5 --strict

probe-env:
	$(PYTHON) scripts/probe_env.py

probe-vram:
	$(PYTHON) scripts/probe_vram.py --amp on

kaggle-payload:
	$(PYTHON) scripts/make_kaggle_payload.py

synthetic:
	$(PYTHON) -m src.synthetic --config $(CONFIG)

# ---- the acceptance path ----------------------------------------------------
smoke:
	$(PYTHON) -m src.synthetic --config $(CONFIG)
	$(PYTHON) -m src.train --task 1 --synthetic --device cpu --override train.epochs=1
	$(PYTHON) -m src.train --task 2 --synthetic --device cpu --override train.epochs=1
	$(PYTHON) -m src.train --task 3 --synthetic --device cpu --override train.epochs=1
	$(PYTHON) -m src.train --task 4 --synthetic --device cpu --override train.epochs=1
	$(PYTHON) -m src.evaluate --synthetic --device cpu
	$(PYTHON) scripts/export_sample_graphs.py --synthetic --n 20
	@echo "smoke OK"

# ---- training ---------------------------------------------------------------
task1:
	$(TRAIN) --task 1 $(OVERRIDE)
task2:
	$(TRAIN) --task 2 $(OVERRIDE)
task3:
	$(TRAIN) --task 3 $(OVERRIDE)
task4:
	$(TRAIN) --task 4 --precompute-text-embeddings $(OVERRIDE)

all-tasks:
	@for s in 42 1337 2024; do \
	  for t in 1 2 3 4; do \
	    $(PYTHON) -m src.train --task $$t --config $(CONFIG) --device $(DEVICE) --seed $$s $(OVERRIDE); \
	  done; \
	done

baselines:
	$(PYTHON) scripts/run_baselines.py --config $(CONFIG) --device $(DEVICE)

evaluate:
	$(PYTHON) -m src.evaluate --config $(CONFIG) --device $(DEVICE)

# C4 runs the seven-mode fusion ablation on Kaggle rather than locally: 21 runs
# at the local 68 ms/row would be ~26 h and would force cutting rows from the
# table. Carries features_*.h5 because Task 3 builds its graphs from those on
# the fly (DataBundle sets graph_dir=None for real data); never mel caches.
kaggle-payload-ablation:
	$(PYTHON) scripts/make_kaggle_payload.py --config $(CONFIG) --no-graphs \
		--include-features --out kaggle_payload_ablation.tar.gz

# ---- -------------------------------------------------------------------
# B2 reads mels_full_{corpus}.h5, not the 256-column pooled cache. Building it
# is the expensive prerequisite for the CNN baseline being worth reporting.
mel-cache:
	$(PYTHON) scripts/build_mel_cache.py --config $(CONFIG) --datasets mtat,fma

# Re-derives tag_vocab.json and musiccaps_tag_vocab.json from the manifests
# already on disk. Cheap, and safe to run any time the splits change.
vocab:
	$(PYTHON) -m src.splits --config $(CONFIG) --vocab-only

thresholds:
	$(PYTHON) scripts/threshold_bootstrap.py --config $(CONFIG) --n-boot 100

# The living report: numbers injected from results/, then structurally checked
# because there is no LaTeX toolchain here to catch a broken macro.
# Two halves, run weeks apart: the sheet is built before the study, the
# analysis after the responses come back. Building the sheet a second time
# would reshuffle the presentation order and orphan the responses, so the
# page target is deliberately separate and not a dependency of the analysis.
human-eval-sheet:
	$(PYTHON) scripts/make_listening_page.py --config $(CONFIG)

human-eval:
	$(PYTHON) scripts/analyse_human_eval.py

report:
	$(PYTHON) scripts/plot_genre_confusion.py --config $(CONFIG)
	$(PYTHON) report/fill_report.py
	$(PYTHON) report/check_tex.py --update

# ---- quality ----------------------------------------------------------------
test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m compileall -q src scripts tests

# ---- cleanup ----------------------------------------------------------------
clean-results:
	rm -rf results/plots results/retrieval_examples results/checkpoints
	rm -f results/metrics.json results/task*_seed*.json results/per_tag_prf.csv results/ablation.csv

clean: clean-results
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ scripts/__pycache__
	rm -rf .pytest_cache runs wandb
	rm -rf data/processed/synthetic data/processed/features.h5 data/processed/mels.h5
	@echo "raw data and data/processed/sample_graphs/ were left alone"
