# GNN–BERT Music Context Understanding

A hybrid BERT + Graph Neural Network system for music context understanding, over
four tasks and four baselines. The whole pipeline runs end to end on synthetic
data in about two minutes on a CPU, so every module can be developed and tested
before the real feature caches exist.

| | Task | Input | Output | Headline metric |
|---|---|---|---|---|
| **T1** | BERT tag classifier | caption / tag text | 50 multi-label tags | macro-F1 |
| **T2** | GNN on structure graphs | audio-only segment graph | **8 FMA genres** (headline) and 50 MTAT tags | accuracy + macro-F1 / macro-F1 |
| **T3** | Cross-attention fusion | graph + text | tags **and** valence/arousal | macro-F1, MAE, R² |
| **T4** | Contrastive dual encoder | graph ↔ caption | retrieval ranking | R@1/5/10, medR, MRR |
| **B1** | Random / majority | — | tags | macro-F1 |
| **B2** | Short-chunk mel CNN | 3 s excerpts, native-resolution log-mel | genres and tags | accuracy + macro-F1 / macro-F1 |
| **B3** | BERT-only | text | tags | macro-F1 |
| **B4** | PCA + MLP | mean-pooled features | tags | macro-F1 |

---

## Results

Four public corpora, 35,984 tracks, 107,952 stored graphs. Every figure below is
generated from `results/*.json`; none is transcribed by hand.

| Task | Result | Reference point |
|---|---|---|
| **T1** tagging, MusicCaps | 0.3707 macro-F1 | masked text; raw captions reach 0.5670 — see below |
| **T2** genre, FMA-small | 43.7% ± 1.8 accuracy, 0.4315 macro-F1 | 12.5% chance; 292,616 parameters |
| **T2** tagging, MagnaTagATune | 0.3737 macro-F1 | +0.0591 over a no-graph control on identical features |
| **T3** fusion, MagnaTagATune | 0.3588 macro-F1 | 7 modes × 3 seeds; 5 within the measurement floor |
| **T3** emotion, DEAM | arousal R² 0.478, valence R² 0.311 | 275 test tracks, jointly trained |
| **T4** retrieval, MusicCaps | R@10 0.0175 | 4.4× chance over a 2,503-clip gallery |

Three findings are worth more than the scores.

**Caption-to-label leakage is large and measurable.** MusicCaps captions are
written *from* the aspect list that also supplies the labels. Two runs identical
except for whether those label terms are masked out of the input differ by
**+0.1963 macro-F1, a 53% relative inflation**. The masked number is the one
reported.

**Which modality dominates depends on the corpus.** The same architecture, run
on two corpora: on MagnaTagATune, whose text channel is title and artist
metadata, the graph carries the task (0.3588 against text's 0.1793); on
MusicCaps, where the text is a written description, the ordering inverts
(0.3095 against 0.1413). Fusion is worth what the second modality contributes,
and no single-corpus study can show that.

**The measurement floor is quantified, then respected.** Resampling the
validation split 100 times moves the tuned test macro-F1 over a range of
**0.0262**. Differences smaller than that are reported as indistinguishable
rather than ranked.

---

## Key design choices

Each of these was measured rather than assumed, and each changed a number that
had already been written down.

**1. The label vocabulary is chosen on the training split.** Ranking tag
frequency over a whole corpus lets test-split annotations decide which labels
exist, before any parameter is trained. On MusicCaps that swaps 7 of the 50
tags. On MagnaTagATune it happens to swap none -- the top-50 set is identical
either way -- which is exactly why it has to be checked rather than argued
about. `make vocab` re-derives both, and both files record `split_used`.

**2. The input text is masked, and the cost of not masking is reported.**
MusicCaps captions are written *from* the aspect list that supplies the labels,
so a model reading the raw caption is doing string matching. Two runs identical
except for masking differ by a large margin in macro-F1, and that gap is
reported as a result rather than quietly avoided. MagnaTagATune ships no
captions at all, so its text channel is `clip_info` metadata -- feeding a
track's own tag string back in would be circular.

**3. Baselines are equalised on compute, not on parameter count.** B2 was
originally width-searched to match the GNN's parameter count, which sounds fair
and is not: convolutional weights are reused at every time-frequency position,
so the same count buys wildly different amounts of computation, and matching it
starved the CNN to an implausible score. Both models now get the same GPU, epoch
cap and early-stopping rule, and every result carries `trainable_params` and
`wall_clock_s`. `target_params` raises if anyone tries to bring it back.

**4. Threshold tuning is treated as a fit, and its variance is measured.**
Per-tag thresholds are tuned on validation and frozen before test is touched --
but MagnaTagATune's validation split is 977 clips from 14 artists, so that fit
is itself noisy. `make thresholds` resamples validation 100 times, re-tunes on
each replicate, and applies each threshold vector to the fixed test split. If
the resulting spread exceeds 0.02 macro-F1, every table must carry the
fixed-0.5 number alongside the tuned one; the verdict is written into
`results/threshold_bootstrap.json` so it cannot be reinterpreted later.

## The report

`report/final_report.tex` is the living document, in IEEEtran two-column form.
There is no LaTeX toolchain in this repository, so it is written to be compiled
elsewhere -- paste it into a fresh Overleaf *IEEE Conference* project, copy the
PNGs from `results/plots/` into `figures/`, and build with pdfLaTeX.

Two scripts stand in for the missing compiler:

```bash
python report/fill_report.py          # inject numbers from results/*.json
python report/check_tex.py --update   # structural checks + page-count estimate
```

The prose contains **no literal numbers** -- only macros, all defined in a
generated block. A result that does not exist yet renders as *pending* and the
script names it, so a stale figure cannot survive a re-run and a missing one
cannot hide.

`results/_synthetic_smoke/` holds artifacts built from the synthetic smoke-test
data. They are kept out of the report pipeline deliberately and should never be
cited as results.

## Installation

**The install order is not optional.** PyTorch Geometric compiles against a
specific torch + CUDA build; installing it before torch, or against a different
CUDA string, produces an import that succeeds and then segfaults during the first
backward pass.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel

# 1) torch FIRST, built against CUDA 12.x (sm_75 / Turing is fully supported)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 2) then PyTorch Geometric
pip install torch-geometric

#    Optional accelerators. Install them ONLY from the wheel index that matches
#    your exact torch+CUDA string, which you can print with:
#        python -c "import torch; print(torch.__version__)"
#    e.g. for 2.14.0+cu126:
#    pip install pyg-lib torch-scatter torch-sparse \
#      -f https://data.pyg.org/whl/torch-2.14.0+cu126.html
#    They are NOT required: SAGEConv and GATv2Conv both run on PyG's pure-torch
#    path, which is what this project uses.

# 3) everything else
pip install -r requirements.txt
```

`make setup` runs exactly these steps.

### System dependencies

| Package | Needed for | Install |
|---|---|---|
| `ffmpeg` | mp3/m4a decoding through librosa, and yt-dlp's clip trimming | `apt install ffmpeg` / `brew install ffmpeg` / `winget install Gyan.FFmpeg` |
| `libsndfile1` | `soundfile`, used to validate that a downloaded clip actually decodes | `apt install libsndfile1` (bundled in the wheel on macOS/Windows) |

### Verified environment

Windows 11, Python 3.13, NVIDIA GTX 1650 Max-Q (4 GB, driver 610.88), CUDA 12.6:
`torch 2.14.0+cu126`, `torch-geometric 2.8.0`, `transformers 5.16.1`,
`numpy 2.5.2`, `pandas 3.0.5`. `requirements.txt` pins what actually resolved
there rather than guessed version numbers.

---

## Quick start

```bash
python scripts/verify_datasets.py         # ALWAYS run this first
python -m src.synthetic                   # contract-compliant fake data
make smoke                                # all four tasks + evaluation, CPU, ~2 min
pytest                                    # unit + smoke tests
```

On Windows the Makefile works under Git Bash with the MinGW make that ships with
it — `mingw32-make smoke PYTHON=.venv/Scripts/python.exe`. If you have no make at
all, `make smoke` is just this sequence:

```bash
python -m src.synthetic
python -m src.train --task 1 --synthetic --device cpu --override train.epochs=1
python -m src.train --task 2 --synthetic --device cpu --override train.epochs=1
python -m src.train --task 3 --synthetic --device cpu --override train.epochs=1
python -m src.train --task 4 --synthetic --device cpu --override train.epochs=1
python -m src.evaluate --synthetic --device cpu
python scripts/export_sample_graphs.py --synthetic --n 20
```

Then, on real data:

```bash
make splits                               # data/splits/*_manifest.csv
make features                             # data/processed/features.h5 (resumable)
make task1 task2 task3 task4 DEVICE=cuda
make baselines
make evaluate
```

---

## Reference hardware

Everything here was developed and measured on a **GTX 1650, 4 GB VRAM** (Turing
TU117, sm_75, **no tensor cores**), an i5-11400H (6c/12t) and 16 GB of RAM. That
budget is not a footnote — it determined several design decisions, and the
project runs within it end to end:

* **AMP everywhere.** `torch.amp.autocast` + `GradScaler` on every training loop.
  On sm_75 the speedup is modest — there are no tensor cores — but the activation
  memory roughly halves, and memory is the binding constraint. bf16 is unsupported
  on this architecture, so fp16 is the only useful autocast dtype.
* **Batch 8 with 4 accumulation steps** for an effective batch of 32. Both are
  config keys (`train.batch_size`, `train.grad_accum_steps`), not constants.
* **`num_workers: 4`, not 12.** The bottleneck is 16 GB of RAM shared with the
  HDF5 cache and a BERT, not CPU threads.
* **Lazy HDF5 reads, per worker.** The feature cache is never loaded whole; each
  worker opens its own read-only handle and reads one track at a time. h5py
  handles do not survive a fork, so `MusicGraphDataset.__getstate__` drops the
  handle rather than pickling a corrupt one into a worker.
* **`float16` feature cache.** Halves the MTAT cache (~1.5 GB → ~750 MB). The
  values are segment-level statistics of dB-scale features, where fp16's precision
  is far below the noise floor of the estimates themselves.
* **`--precompute-text-embeddings` for Task 4.** Contrastive learning wants large
  batches because the in-batch negatives *are* the signal. 512 BERT forward passes
  do not fit in 4 GB; 512 rows sliced out of a precomputed `[N, d]` CPU matrix do.
* **CPU is a first-class device.** Every script takes `--device {cuda,cpu}` and
  runs correctly on either; the demo notebook finishes on CPU in under 2 minutes.

---

## The frozen data contract

Every task reads the same structures, so changing any of them invalidates the
cached features, the stored graphs and every result derived from them. The
shapes below are asserted at load time.

### Segment graph — `torch_geometric.data.Data`

| Field | Shape | Dtype | Meaning |
|---|---|---|---|
| `x` | `[num_nodes, 96]` | float32 | segment features, canonical order below |
| `edge_index` | `[2, num_edges]` | int64 | COO, bidirectional |
| `edge_attr` | `[num_edges, 2]` | float32 | `[is_temporal, cosine_sim]` |
| `y_tags` | `[1, num_tags]` | float32 | multi-hot; `-1` = label absent for this dataset |
| `y_genre` | `[1]` | int64 | `-1` if unavailable |
| `y_valence` | `[1]` | float32 | `nan` if unavailable |
| `y_arousal` | `[1]` | float32 | `nan` if unavailable |
| `track_id`, `artist_id`, `dataset`, `split`, `text` | — | str | metadata; `artist_id` drives the leakage assertion |

### Canonical 96-dim node feature order

```
[  0: 16]  mel-band mean over 16 pooled mel bands
[ 16: 32]  mel-band std  over 16 pooled mel bands
[ 32: 52]  MFCC mean               (20)
[ 52: 72]  MFCC std                (20)
[ 72: 84]  chroma mean             (12)
[ 84: 91]  spectral contrast mean   (7)
[ 91: 96]  centroid, bandwidth, rolloff, zcr, rms  (5)
                                        ------ 96
```

Asserted at build time in both `audio_features.segment_features` and
`graph_builder.build_segment_graph`.

### The missing-label sentinel rule — the most important rule here

`-1` for an absent tag vector, `nan` for an absent regression target, and
**every loss masks them out**. No track carries both MTAT tags and DEAM
valence/arousal, so an unmasked BCE would train the model that every DEAM track
has fifty confident negative tags. The loss curve would look healthy and macro-F1
would collapse. `fusion_model.masked_multitask_loss` is the single place this is
handled, and `tests/test_smoke.py` asserts that sentinel rows receive exactly zero
gradient.

### Manifest schema — `data/splits/{dataset}_manifest.csv`

```
track_id, artist_id, audio_path, text, split, y_genre, y_tags (json list),
y_valence, y_arousal, duration_s
```

---

## Datasets

Configure the paths in `config.yaml → datasets:`; nothing is hardcoded. The
download scripts under `scripts/` are documented and idempotent, but **nothing is
downloaded automatically** — run them deliberately.

| Dataset | Path | Nominal | Notes |
|---|---|---|---|
| MagnaTagATune | `data/raw/mtat/` | 25,863 clips, 188 tags | folder split `0–b` train / `c` val / `d–f` test (artist-disjoint) |
| FMA-small | `data/raw/fma/` | 8,000 × 30 s, 8 genres | official `set.split` column |
| MusicCaps | `data/raw/musiccaps/` | 5,521 rows | **9% unavailable after recovery** (was ~50%) — see below |
| DEAM | `data/raw/deam/` | 1,802 excerpts | valence/arousal on a 1–9 scale; artist-grouped 70/15/15 |
| Lakh MIDI Clean | `data/raw/lmd_clean/` | ~17k MIDI files | chord-estimator validation only, not a training input |

### MusicCaps needs special handling

MusicCaps ships captions, not audio; each row points at a YouTube video, and
roughly half of them are now deleted, private or geo-blocked. **The source CSV is
never modified.** Instead:

* `splits.build_musiccaps_manifest()` inner-joins the CSV against files that exist
  *and decode*. `yt-dlp` leaves behind 0-byte and truncated files that pass an
  existence check, so each candidate is opened with `soundfile` and its duration
  checked against the nominal 10 s.
* `data/splits/musiccaps_download_log.csv` gets one row per original ytid with a
  status in `{ok, missing, corrupt, wrong_duration}`.
* `data/splits/musiccaps_manifest.csv` holds only the usable rows.
* Survival is reported per split using `is_audioset_eval`, and **the eval survivor
  count is the Task 4 retrieval gallery size**, recorded in `results/metrics.json`.
  R@10 out of 1,400 and R@10 out of 2,858 are different claims.

On the reference machine, **after running the recovery pass**
(`scripts/download_musiccaps.py --retry-failed`): **5,043 of 5,521 rows usable
(91.3%)** — 2,409/2,663 train and
**2,634/2,858 eval**, so the retrieval gallery is **2,634**.

Before recovery it was 2,781 usable (50.4%) with a gallery of 1,481. The retry
pass re-attempted the 2,740 previously-failed ids and succeeded on 82.4% of them,
because most earlier failures were transient (rate limiting and timeouts) rather
than genuinely deleted videos. **Report both numbers**: which one you have
changes R@K materially, and a gallery that nearly doubled is not a footnote.

---

## Design decisions worth defending

**Similarity edges use k-NN, not a cosine threshold τ.** A single global τ gives
isolated nodes on a through-composed track and a near-clique on a loop-based one,
so the topology ends up encoding "how repetitive is this track" rather than "which
segments belong together", and the GNN's receptive field varies wildly across a
batch. Fixed `graph.knn_k` keeps message passing comparable; the cosine value is
kept in `edge_attr` so the model can still discount weak neighbours.

**Two GNN layers, not six.** These graphs have 4–32 nodes and a diameter of maybe
4–6 hops. At three or four rounds of message passing every node sees the whole
graph and the node states converge — oversmoothing, which shows up as a *drop* in
macro-F1 that is easily mistaken for underfitting. `GNNEncoder` warns above four.

**Separate learning rates for BERT and everything else.** A single shared LR
reliably produces a model that ignores one branch: at 1e-3 the pretrained encoder
is destroyed within a few hundred steps; at 2e-5 the randomly initialised GNN
never leaves its initialisation. Every model exposes `param_groups(lr_bert,
lr_head)`, BERT is frozen for `bert.freeze_epochs`, then the top
`bert.unfreeze_top_n_layers` are unfrozen and the optimiser is rebuilt.

**Thresholds are tuned on validation and frozen.** `metrics.tune_thresholds` is
called on the validation scores at the end of training, stored in the result JSON
with `threshold_source: "val"`, and only then applied to test. The test split is
read exactly once, after both the model and the thresholds are final.

**Normalisation statistics come from train only.** `compute_norm_stats` raises if
asked for any other split, and `train.DataBundle` refuses to run against a
`norm_stats.json` that is not marked `"train"`.

**Accuracy is never reported for multi-label tagging.** With 50 tags where the
median track carries ~4, an all-zeros predictor scores >90% element accuracy and
0.00 macro-F1. `baselines.majority_tag_baseline` computes that number *once*,
under the key `element_accuracy_do_not_report`, precisely to make the point in the
report. Everywhere else: macro-F1, micro-F1, AUC-PR.

**What goes into `Xtext`, and why it is not the labels.** MTAT ships **no
captions or lyrics** - only tags. Feeding a track's own tags in as text to
predict those same tags is degenerate, and MusicCaps is subtler but worse: its
captions are *written from* the aspect list, so the raw caption contains its own
labels almost verbatim. The configuration is locked as:

| Run | Dataset | `data.text_source` | Purpose |
|---|---|---|---|
| Task 1 headline | MusicCaps | `caption_masked` | the main Task 1 number |
| Task 1 leakage demo | MusicCaps | `caption_raw` | quantifies the inflation |
| Task 1 secondary | MTAT | `metadata` | non-circular, weak signal |
| Task 3 primary | MTAT | `metadata` | satisfies the PDF "results on MTAT" |
| Task 3 secondary | MusicCaps | `caption_masked` | rich-text domain |

`splits.strip_aspect_terms` removes each aspect phrase *and* its content tokens
with inflections (including consonant doubling, so `drum` also catches
`drumming` and `drummer`), longest phrases first. On the real corpus it fires on
**2,779 of 2,781 captions** (mean 10.8 aspects each) and retains **72% of the
words** - 49.5 to 35.5 words on average, with no caption emptied. Variants live
in `data/splits/{dataset}_text_variants.csv`; the manifest schema stays frozen at
its ten columns, and `DataBundle` swaps `text` at load time.

The payoff is a comparison rather than a single number: fusion gain should scale
with text informativeness - large on MusicCaps captions, small on MTAT metadata.

**Splits are artist-disjoint, and it is asserted.** `assert_no_leakage` runs at the
top of every training script and raises — not warns — if any track id or artist id
appears in two splits. MTAT is drawn from Magnatune, where one artist contributes
many clips from the same album; a random clip split trains and tests on the same
recording session.

This assertion immediately found a real problem. **MagnaTagATune's canonical
hex-folder split is not actually artist-disjoint**: 57 artists have clips in more
than one directory. `splits.enforce_artist_disjoint` (on by default) repairs it by
moving each such artist wholesale into whichever split already holds most of its
clips, which relocates **3,465 of 21,361 clips — 16.2% of the corpus**. Set
`splits.enforce_artist_disjoint: false` to reproduce the unrepaired literature
split, and say so in the report if you do; numbers from it are optimistic by an
unknown margin.

**The rewired-graph control.** `graph_builder.rewire_edges` does degree-preserving
double-edge swaps. It appears in the ablation table as its own row. If performance
survives rewiring, the topology was never carrying signal and the GNN is a pooled-
feature MLP with extra steps — which is worth knowing.

---

## Repository layout

```
gnn-bert-music-context/
  README.md  requirements.txt  config.yaml  Makefile  pytest.ini  .gitignore
  data/
    raw/          the five corpora (read-only inputs, git-ignored)
    processed/    feature caches (git-ignored) + sample_graphs/ (COMMITTED)
    splits/       manifests, the MusicCaps download log, the tag vocabulary
  notebooks/
    eda.ipynb              tag long tail, durations, co-occurrence, survival table
    demo_context.ipynb     one audio file -> tags + V/A + top-3 captions, <2 min CPU
  src/
    utils.py            seeding, config, device, AMP helpers, VRAM logging
    metrics.py          ALL metrics -- single source of truth
    audio_features.py   96-dim features + the resumable HDF5 cache
    chords.py           chord estimation + Lakh MIDI validation
    graph_builder.py    segment / chord / hetero graphs, rewiring, rendering
    splits.py           manifests, synonym merging, the leakage assertion
    datasets.py         torch Datasets, PyG loaders, alternating loader
    synthetic.py        contract-compliant fake data
    bert_encoder.py     T1 / B3
    gnn_model.py        T2 + the hetero variant
    cnn_baseline.py     B2 short-chunk CNN (NOT parameter-matched -- see A7.2)
    baselines.py        B1, B4
    fusion_model.py     T3 + masked_multitask_loss
    contrastive.py      T4
    train.py            unified entry point for all four tasks
    evaluate.py         regenerates every table and plot
    attention_viz.py    BERT + GAT attention figures, case studies
    human_eval.py       listening sheets, Krippendorff alpha, control checks
  scripts/
    verify_datasets.py  run this first
    download_*.sh/.py   documented, idempotent; not run automatically
    export_sample_graphs.py, run_baselines.py
  results/
    metrics.json  plots/  retrieval_examples/  checkpoints/ (git-ignored)
  report/
    final_report.pdf
```

`data/processed/sample_graphs/` is the one binary directory that **is**
committed: 20 real graphs, so the repository carries runnable examples without
requiring the full 30 GB of source audio.

---

## Usage

```bash
# train
python -m src.train --task {1,2,3,4} --config config.yaml \
                    [--synthetic] [--seed N] [--device cuda|cpu] \
                    [--override key=value ...] [--dry-run] \
                    [--precompute-text-embeddings]

# examples
python -m src.train --task 2 --override gnn.conv=gatv2 gnn.readout=attention
python -m src.train --task 4 --precompute-text-embeddings --override contrastive.batch_size=512
python -m src.train --task 3 --override fusion.mode=gated train.epochs=15

# everything, over the three configured seeds
make all-tasks DEVICE=cuda

# regenerate every table and plot from the saved checkpoints
python -m src.evaluate --device cuda
```

Any config key can be overridden with dotted notation; CLI always wins over
`config.yaml`.

### What `evaluate` produces

`results/metrics.json` plus `results/plots/`:
F1-vs-epoch curves · per-tag PRF table and bar chart · per-tag error decomposition
· ablation table over all seven fusion modes plus the rewired control · t-SNE
coloured by genre and by mood quadrant, each annotated with its k-NN probe and
silhouette · retrieval R@K in both directions with the gallery size · S_graph for
the trained model against its rewired control · seed-aggregated mean ± sd · five
BERT attention heatmaps · three case studies (at least one a failure) ·
`results/retrieval_examples/` with ten examples, at least two of them failures.

---

## Tests

```bash
pytest                      # everything
pytest -m "not slow"        # unit tests only, ~10 s
pytest tests/test_smoke.py  # all four tasks, 1 epoch, CPU
```

* `test_metrics.py` — hand-computed macro/micro F1, AUC-PR, retrieval R@K,
  `S_graph`, plus sentinel-masking cases.
* `test_graph_builder.py` — 96-dim features, constant k-NN degree, no isolated
  nodes, self-loops, `edge_attr` shape, degree-preserving rewiring.
* `test_splits.py` — the leakage assertion in both directions, synonym merging.
* `test_smoke.py` — all four tasks train one epoch on CPU with finite losses and
  non-`nan` metrics; the masked loss provably ignores `-1` and `nan`.

---

## Troubleshooting

**`OSError: ... torch_sparse ... undefined symbol`** — PyG companions built for a
different torch. Uninstall them; the project does not need them.

**CUDA OOM on the 1650** — lower `train.batch_size` and raise
`train.grad_accum_steps` to keep the effective batch; confirm `amp: true`; for
Task 4 use `--precompute-text-embeddings`. `log_vram()` prints the peak after each
epoch.

**`KeyError: no cached features for ...`** — run `make features`. Extraction is
resumable, so re-running after an interruption only does what is left.

**`AssertionError: LEAKAGE:`** — working as intended. An artist appears in two
splits; fix the manifest rather than the assertion.

**Tokenizer fails to load** — `transformers` 5.x requires a fast tokenizer. Some
older Hub checkpoints (e.g. `prajjwal1/bert-tiny`) ship only a slow one; use
`distilbert-base-uncased` or `bert-base-uncased`.

**`make: command not found` on Windows** — use `mingw32-make` (bundled with Git
for Windows) or run the target's commands directly; they are all plain
`python -m ...` invocations.

**MusicCaps survival looks low** — it is. ~50% is expected. Check
`data/splits/musiccaps_download_log.csv` for the per-status breakdown, and
`scripts/download_musiccaps.py --retry-failed` for a recovery pass from a
different network.

---

## What is not in this repository

| Excluded | Why |
|---|---|
| `data/raw/` and the corpus directories by name | ~30 GB of audio; MusicCaps clips are YouTube-sourced and not redistributable |
| `data/processed/` (except `sample_graphs/`) | 1.4 GB of graphs + 2.5 GB of HDF5 caches, all regenerable |
| `results/checkpoints/`, `results/_synthetic_smoke/` | model weights; the synthetic set must never reach a report |
| `logs/`, `*.tar.gz` | run logs and rebuildable upload payloads |
| every credential pattern | `.env`, `kaggle.json`, `*.pem`, `*.key`, `id_rsa*`, `.netrc`, `.aws/`, `.huggingface/`, … |

`data/processed/sample_graphs/` is the **one** deliberate binary exception: 20
real `.pt` graphs, re-included after the `*.pt` rule so the repository carries
runnable examples. Roughly 35 MB is tracked in total, mostly manifest CSVs.

---

## Authors

Utsha Basak · Mohammad Tanvir Hossain · Nabil Mahmud
Department of Computer Science and Engineering, BRAC University

## License

The code in this repository is released under the MIT License; see `LICENSE`.

The corpora are not redistributed here and remain under their own terms:
MagnaTagATune, FMA, DEAM, MusicCaps and Lakh MIDI Clean each carry separate
licences, and MusicCaps in particular ships YouTube identifiers rather than
audio. See `scripts/download_*.sh` and `data/raw/README.md` for how to obtain
them.
