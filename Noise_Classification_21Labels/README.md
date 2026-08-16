# 21-label noise classification at 16 kHz

This directory is an independent adaptation of `U_FFIA27K_audio` for
`21_labels_dataset`. The original fish-feeding source directory is unchanged.

## What this pipeline does

- Reads the authoritative `train_single`, `validation_single`, and `test_single`
  manifests; it never re-splits the noise sources.
- Rebuilds WAV paths from `sample_id`, so stale absolute paths stored in the
  manifests are ignored.
- Maps the original AudioSet label indices to 21 contiguous model outputs using
  `selected_labels.csv`.
- Trains a multi-label model with `BCEWithLogitsLoss` and optional class weights.
- Uses 16 kHz audio, random 4-second train crops, and deterministic 4-second
  sliding windows with a 2-second hop for validation, test, and inference.
- Optionally creates controlled time-varying SNR mixtures from `clean` and
  `oracle_noise` during training.
- Reports mAP, macro/micro F1, per-label metrics, and metrics for each SNR band.

## Expected dataset layout

```text
21_labels_dataset/
├── selected_labels.csv
├── train_single/
│   ├── manifest.csv
│   ├── mixture/
│   ├── clean/
│   └── oracle_noise/
├── validation_single/
└── test_single/
```

## Run on Marimo

Place the dataset at `/marimo/21_labels_dataset`. If it is elsewhere, edit
`dataset_splitter.dataset_path` in `config/train_config.json` or set the
`NOISE_DATASET_PATH` environment variable.

Install dependencies in a Marimo terminal:

```bash
cd /marimo/Capstone_2026_Fish_Feeding_Intensity/Noise_Classification_21Labels
python -m pip install -r requirements.txt
```

Start training from a terminal:

```bash
python main.py --config config/train_config.json --check-data
python main.py --config config/train_config.json --device cuda
```

Or run it from a Marimo cell:

```python
from pathlib import Path
import sys

project = Path("/marimo/Capstone_2026_Fish_Feeding_Intensity/Noise_Classification_21Labels")
sys.path.insert(0, str(project))

from main import run_training

result = run_training(
    str(project / "config" / "train_config.json"),
    device_name="cuda",
)
result["checkpoint_path"]
```

The best model and reports are written to:

```text
checkpoint/<backbone>/
├── audio_best.pt
├── history.csv
├── learning_curves.png
├── summary.csv
├── summary.json
├── test_per_label.csv
└── classification_report_test.txt
```

## Sliding-window inference

```bash
python inference.py /marimo/example.wav \
  --config config/train_config.json \
  --checkpoint checkpoint/Cnn14MobileV2/audio_best.pt \
  --device cuda
```

This creates a CSV containing the probability of every label in every time
window, plus a JSON clip-level summary.

## Important configuration knobs

- `batch_size`: start with 32; reduce to 16 or 8 if CUDA runs out of memory.
- `num_workers`: `-1` auto-selects up to eight workers. Use `0` if the Marimo
  runtime has multiprocessing issues.
- `cache_audio`: keep `false` for this 15 GB dataset.
- `dynamic_snr_enabled`: enables clean/noise on-the-fly mixing.
- `dynamic_snr_probability`: fraction of train crops receiving dynamic SNR.
- `threshold`: initial multi-label decision threshold. Tune it on validation
  predictions after the first training run.
- `profile_model`: disabled by default because FLOP profiling adds startup time.

The manifest stores weak clip-level labels, not event timestamps. Per-window
outputs therefore indicate model confidence over time but are not supervised
on exact noise onset/offset boundaries.
