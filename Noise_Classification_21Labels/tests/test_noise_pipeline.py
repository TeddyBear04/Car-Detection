import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AudioFeaturesConfig, SplitterConfig
from dataset import NoiseDataLoaderManager, sliding_window_starts
from utils.evaluate import aggregate_windows, compute_multilabel_metrics


class NoisePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        with (self.root / "selected_labels.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model_index", "original_index", "mid", "display_name"])
            writer.writerows([[0, 10, "/m/a", "A"], [1, 20, "/m/b", "B"], [2, 30, "/m/c", "C"]])

        for split in ("train_single", "validation_single", "test_single"):
            split_path = self.root / split
            for signal_type in ("mixture", "clean", "oracle_noise"):
                (split_path / signal_type).mkdir(parents=True, exist_ok=True)
            rows = []
            for index, labels in enumerate(([10], [20, 30])):
                sample_id = f"{split}_{index:02d}"
                duration = 1.5 + index
                samples = int(16_000 * duration)
                time = np.arange(samples, dtype=np.float32) / 16_000
                clean = 0.1 * np.sin(2 * np.pi * 300 * time)
                noise = 0.03 * np.sin(2 * np.pi * 900 * time)
                for signal_type, waveform in (
                    ("clean", clean),
                    ("oracle_noise", noise),
                    ("mixture", clean + noise),
                ):
                    sf.write(split_path / signal_type / f"{sample_id}.wav", waveform, 16_000)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "label_indices": str(labels),
                        "target_snr_db": "5.0",
                        "duration_seconds": str(duration),
                    }
                )
            with (split_path / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_id", "label_indices", "target_snr_db", "duration_seconds"],
                )
                writer.writeheader()
                writer.writerows(rows)

        self.dataset_config = SplitterConfig(
            dataset_path=str(self.root),
            dynamic_snr_enabled=True,
            dynamic_snr_probability=1.0,
        )
        self.audio_config = AudioFeaturesConfig(
            sample_rate=16_000,
            clip_seconds=1.0,
            inference_hop_seconds=0.5,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmax=8_000,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_manifest_loader_and_dynamic_snr(self) -> None:
        manager = NoiseDataLoaderManager(
            self.dataset_config,
            self.audio_config,
            batch_size=2,
            num_workers=0,
            cache_audio=False,
            pin_memory=False,
            classes_num=3,
        )
        train_item = manager.datasets["train"][0]
        self.assertEqual(tuple(train_item["waveform"].shape), (16_000,))
        self.assertEqual(tuple(train_item["target"].shape), (3,))
        self.assertTrue(bool(train_item["dynamic_snr"]))
        self.assertTrue(bool(torch.isfinite(train_item["waveform"]).all()))
        self.assertGreater(len(manager.datasets["val"]), len(manager.datasets["val"].records))
        self.assertEqual(manager.label_names, ["A", "B", "C"])

    def test_windowing_and_multilabel_metrics(self) -> None:
        self.assertEqual(sliding_window_starts(25, 10, 6), [0, 6, 12, 15])
        target = np.asarray([[1, 0, 0], [0, 1, 1]], dtype=np.float32)
        probability = np.asarray([[0.9, 0.1, 0.2], [0.1, 0.8, 0.9]], dtype=np.float32)
        metrics = compute_multilabel_metrics(target, probability, 0.5, ["A", "B", "C"])
        self.assertAlmostEqual(metrics["mAP"], 1.0)
        self.assertAlmostEqual(metrics["f1_macro"], 1.0)
        ids, clip_probability, _, _ = aggregate_windows(
            ["one", "one", "two"],
            np.ones((3, 3), dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
            np.asarray([0.0, 0.0, 5.0], dtype=np.float32),
        )
        self.assertEqual(ids, ["one", "two"])
        self.assertEqual(clip_probability.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
