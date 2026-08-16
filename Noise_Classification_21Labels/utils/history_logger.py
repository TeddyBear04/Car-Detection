from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class HistoryLogger:
    def __init__(self, log_dir: str, label_names: Sequence[str], threshold: float = 0.5) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.label_names = list(label_names)
        self.threshold = threshold
        self.history_path = self.log_dir / "history.csv"
        self.headers = [
            "epoch",
            "train_loss",
            "train_mAP",
            "train_macro_f1",
            "train_micro_f1",
            "val_loss",
            "val_mAP",
            "val_macro_f1",
            "val_micro_f1",
            "val_subset_accuracy",
            "is_best",
        ]
        with self.history_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.headers).writeheader()

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_statistics: Dict[str, Any],
        val_statistics: Dict[str, Any],
        is_best: bool,
    ) -> None:
        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_mAP": f"{train_statistics['mAP']:.6f}",
            "train_macro_f1": f"{train_statistics['f1_macro']:.6f}",
            "train_micro_f1": f"{train_statistics['f1_micro']:.6f}",
            "val_loss": f"{val_statistics['loss']:.6f}",
            "val_mAP": f"{val_statistics['mAP']:.6f}",
            "val_macro_f1": f"{val_statistics['f1_macro']:.6f}",
            "val_micro_f1": f"{val_statistics['f1_micro']:.6f}",
            "val_subset_accuracy": f"{val_statistics['subset_accuracy']:.6f}",
            "is_best": int(is_best),
        }
        with self.history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.headers)
            writer.writerow(row)
        if is_best:
            self.save_per_label_metrics("validation_per_label_best.csv", val_statistics)

    def save_per_label_metrics(self, filename: str, statistics: Dict[str, Any]) -> None:
        path = self.log_dir / filename
        rows = []
        for index, label in enumerate(self.label_names):
            matrix = statistics["confu_matrix"][index]
            rows.append(
                {
                    "model_index": index,
                    "label": label,
                    "average_precision": statistics["average_precision"][index],
                    "auc": statistics["auc"][index],
                    "tn": int(matrix[0, 0]),
                    "fp": int(matrix[0, 1]),
                    "fn": int(matrix[1, 0]),
                    "tp": int(matrix[1, 1]),
                }
            )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _summary_values(prefix: str, statistics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            f"{prefix}_loss": statistics["loss"],
            f"{prefix}_mAP": statistics["mAP"],
            f"{prefix}_macro_f1": statistics["f1_macro"],
            f"{prefix}_micro_f1": statistics["f1_micro"],
            f"{prefix}_macro_auc": statistics["macro_auc"],
            f"{prefix}_hamming_accuracy": statistics["hamming_accuracy"],
            f"{prefix}_subset_accuracy": statistics["subset_accuracy"],
            f"{prefix}_clips": statistics["num_clips"],
            f"{prefix}_windows": statistics["num_windows"],
        }

    def save_summary(
        self,
        training_time: float,
        inference_time_ms: float,
        best_epoch: int,
        val_statistics: Dict[str, Any],
        test_statistics: Dict[str, Any],
    ) -> None:
        summary = {
            "training_time_seconds": training_time,
            "inference_time_ms_per_window": inference_time_ms,
            "best_epoch": best_epoch,
            **self._summary_values("val", val_statistics),
            **self._summary_values("test", test_statistics),
        }
        with (self.log_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)

        details = {
            "summary": summary,
            "validation_snr_metrics": val_statistics["snr_metrics"],
            "test_snr_metrics": test_statistics["snr_metrics"],
            "threshold": self.threshold,
        }
        with (self.log_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(details, handle, indent=2, ensure_ascii=False)
        self.save_per_label_metrics("test_per_label.csv", test_statistics)
        (self.log_dir / "classification_report_test.txt").write_text(
            test_statistics["message"], encoding="utf-8"
        )
        logger.info("Saved training summary to %s", self.log_dir)

    def plot_history(self) -> None:
        rows = []
        with self.history_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [int(row["epoch"]) for row in rows]
        figure, axes = plt.subplots(1, 3, figsize=(17, 5))
        pairs = [
            ("Loss", "train_loss", "val_loss"),
            ("mAP", "train_mAP", "val_mAP"),
            ("Macro F1", "train_macro_f1", "val_macro_f1"),
        ]
        for axis, (title, train_key, val_key) in zip(axes, pairs):
            axis.plot(epochs, [float(row[train_key]) for row in rows], label="train")
            axis.plot(epochs, [float(row[val_key]) for row in rows], label="validation")
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.3)
            axis.legend()
        figure.suptitle("21-label noise classification")
        figure.tight_layout()
        figure.savefig(self.log_dir / "learning_curves.png", dpi=150)
        plt.close(figure)
