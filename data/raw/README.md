# data/raw -- the corpora, which are NOT in this repository

About 30 GB of audio lives here and is deliberately gitignored. This file is
committed so that a fresh clone has the directory structure the project expects.

Expected layout (paths are configurable in `config.yaml` under `datasets:`):

    data/raw/
      mtat/
        audio/                  MagnaTagATune mp3s, in hex folders 0-f
        annotations_final.csv
        clip_info_final.csv
      fma/
        fma_small/              FMA-small mp3s, in numbered folders
        fma_metadata/
      musiccaps/
        audio/                  10 s clips named <ytid>_<start>_<end>.mp3
        musiccaps-public.csv    NEVER modified by this project
      deam/
        audio/
        annotations/
        metadata/
      lmd_clean/                Lakh MIDI Clean, used only to validate chords

`python scripts/verify_datasets.py` reports what is present, what is missing and
what fails to decode. MusicCaps ships YouTube ids rather than audio, so
`scripts/download_musiccaps.py` fetches it; expect roughly 12% attrition, which
is not random and is discussed in the report.

Nothing in this directory is ever written to by the pipeline.
