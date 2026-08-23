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
- Reports mAP, macro/micro F1, accuracy, per-label metrics, and metrics for each
  SNR band.

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
├── test_snr_metrics.csv
├── validation_snr_metrics.csv
├── snr_metrics.png
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

## Accuracy metrics

This is a multi-label task, so "accuracy" is ambiguous. Two definitions are
reported everywhere (per-epoch log, `history.csv`, `summary.csv`/`summary.json`,
`learning_curves.png`, and per SNR band):

- `hamming_accuracy` — fraction of the 21 label decisions that are correct,
  averaged over every clip. This is the metric plotted and printed as `acc`.
  Because the labels are sparse, a model that predicts all-zero already scores
  around 0.9, so read it alongside macro F1 rather than on its own.
- `subset_accuracy` — fraction of clips where all 21 labels are exactly right
  (exact-match ratio). This is very harsh at 21 labels and typically sits at or
  near 0.0 for a long time; it is logged as `exact-match`, not as `acc`.

`test_per_label.csv` and `validation_per_label_best.csv` also carry an
`accuracy` column giving the per-label correctness rate. Its mean over the 21
labels equals `hamming_accuracy`.

Both are threshold-dependent: they change when you tune `threshold`. Only `mAP`
and `auc` are threshold-free.

## Results by SNR band

Every evaluation is also broken down by the clip's `target_snr_db`, which is the
main way to see how the model degrades as noise increases. Each band reports
`samples`, `mAP`, `macro_auc`, `macro_f1`, `micro_f1`, `precision_macro`,
`recall_macro`, `hamming_accuracy`, and `subset_accuracy`.

Output lands in three places:

- `test_snr_metrics.csv` and `validation_snr_metrics.csv` — one row per band.
- `snr_metrics.png` — grouped bar chart of mAP / macro F1 / micro F1 / accuracy.
- `summary.json` — same numbers under `test_snr_metrics` and
  `validation_snr_metrics`.

The end of training also prints the table:

```text
        band   clips        mAP  macro-AUC   macro-F1   micro-F1  precision     recall        acc      exact
      [-5,0]    1301     0.3124     0.6708     0.3245     0.3309     0.2671     0.4237     0.7457     0.0049
      [5,10]    1316     0.6087     0.8668     0.5375     0.5394     0.4139     0.7733     0.7964     0.0000
     [15,20]    1289     0.8194     0.9483     0.6134     0.6145     0.4614     0.9288     0.8214     0.0199
```

The bands are configurable via `snr_bands` in `config/train_config.json`. Bounds
are inclusive on both ends. The defaults match how `21_labels_dataset` was
generated — target SNR is drawn from three separated clusters, so these three
bands cover every clip exactly once (verified on all three splits: 5992 + 6234 +
6005 = 18231 train clips, with none left over). If you change the bands and
leave clips outside all of them, a warning naming the uncovered count is logged
rather than silently dropping them.

## Important configuration knobs

- `batch_size`: start with 32; reduce to 16 or 8 if CUDA runs out of memory.
- `num_workers`: `-1` auto-selects up to eight workers. Use `0` if the Marimo
  runtime has multiprocessing issues.
- `cache_audio`: keep `false` for this 15 GB dataset.
- `dynamic_snr_enabled`: enables clean/noise on-the-fly mixing.
- `dynamic_snr_probability`: fraction of train crops receiving dynamic SNR.
- `threshold`: initial multi-label decision threshold. Tune it on validation
  predictions after the first training run.
- `monitor`: metric that drives best-checkpoint selection and early stopping.
  One of `macro_f1`, `mAP`, `hamming_accuracy`, `subset_accuracy`, `loss`.
  Avoid `subset_accuracy` here: it stays flat at 0.0 early in training, so
  early stopping would fire before the model learns anything.
- `profile_model`: disabled by default because FLOP profiling adds startup time.

The manifest stores weak clip-level labels, not event timestamps. Per-window
outputs therefore indicate model confidence over time but are not supervised
on exact noise onset/offset boundaries.
