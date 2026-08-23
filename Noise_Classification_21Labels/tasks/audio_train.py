from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils import (
    AudioEvaluator,
    EarlyStopping,
    HistoryLogger,
    InferenceTimer,
    MultiLabelBCELoss,
)
from utils.evaluate import compute_multilabel_metrics

logger = logging.getLogger(__name__)


def _format_snr_table(snr_metrics: Dict[str, Dict[str, Any]]) -> str:
    """Render the per-SNR breakdown as a fixed-width table for the log."""
    if not snr_metrics:
        return " (no SNR bands matched any clip)"
    columns = [
        ("mAP", "mAP"),
        ("macro-AUC", "macro_auc"),
        ("macro-F1", "macro_f1"),
        ("micro-F1", "micro_f1"),
        ("precision", "precision_macro"),
        ("recall", "recall_macro"),
        ("acc", "hamming_accuracy"),
        ("exact", "subset_accuracy"),
    ]
    header = f"\n  {'band':>10s} {'clips':>7s}" + "".join(f" {title:>10s}" for title, _ in columns)
    lines = [header, "  " + "-" * (len(header) - 3)]
    for name, values in snr_metrics.items():
        row = f"  {name:>10s} {values['samples']:>7d}"
        row += "".join(f" {values[key]:>10.4f}" for _, key in columns)
        lines.append(row)
    return "\n".join(lines)


class BaseTrainer:
    def train(self, train_loader: Any, val_loader: Any, test_loader: Any, max_epoch: int) -> Dict[str, Any]:
        raise NotImplementedError


class AudioTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        ckpt_dir: str,
        label_names: Sequence[str],
        threshold: float = 0.5,
        monitor: str = "macro_f1",
        early_stopping: bool = True,
        patience: int = 15,
        delta: float = 0.0,
        pos_weight: torch.Tensor | None = None,
        clip_samples: int = 64_000,
        train_config_path: str = "config/train_config.json",
        snr_bands: Sequence[tuple[str, float, float]] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.label_names = list(label_names)
        self.threshold = threshold
        self.monitor = monitor
        self.clip_samples = clip_samples
        self.loss_fn = MultiLabelBCELoss(pos_weight=pos_weight).to(device)
        self.evaluator = AudioEvaluator(
            model=model,
            label_names=self.label_names,
            threshold=threshold,
            loss_fn=self.loss_fn,
            window_reduction="mean",
            snr_bands=snr_bands,
        )
        self.early_stopper = (
            EarlyStopping(patience=patience, delta=delta, verbose=True) if early_stopping else None
        )
        self.history = HistoryLogger(str(self.ckpt_dir), self.label_names, threshold=self.threshold)
        config_path = Path(train_config_path)
        if config_path.is_file():
            shutil.copy2(config_path, self.ckpt_dir / "train_config.json")
        (self.ckpt_dir / "labels.json").write_text(
            json.dumps(self.label_names, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _monitor_value(self, statistics: Dict[str, Any]) -> float:
        if self.monitor == "macro_f1":
            return float(statistics["f1_macro"])
        if self.monitor == "mAP":
            return float(statistics["mAP"])
        if self.monitor == "hamming_accuracy":
            return float(statistics["hamming_accuracy"])
        if self.monitor == "subset_accuracy":
            return float(statistics["subset_accuracy"])
        if self.monitor == "loss":
            return -float(statistics["loss"])
        raise ValueError(f"Unsupported monitor: {self.monitor}")

    def _save_checkpoint(self, epoch: int, score: float) -> Path:
        path = self.ckpt_dir / "audio_best.pt"
        torch.save(
            {
                "epoch": epoch,
                "monitor": self.monitor,
                "monitor_score": score,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "label_names": self.label_names,
                "threshold": self.threshold,
            },
            path,
        )
        logger.info("Saved best checkpoint: %s", path)
        return path

    def _train_epoch(self, train_loader: Any, epoch: int) -> tuple[float, Dict[str, Any]]:
        self.model.train()
        loss_sum = 0.0
        sample_count = 0
        probabilities = []
        targets = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", dynamic_ncols=True)
        for batch in progress:
            waveform = batch["waveform"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            output = self.model(waveform)
            loss = self.loss_fn(output, {"target": target})
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            batch_size = waveform.size(0)
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size
            probabilities.append(torch.sigmoid(output["clipwise_output"]).detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        probability = np.concatenate(probabilities, axis=0)
        target = np.concatenate(targets, axis=0)
        statistics = compute_multilabel_metrics(
            target,
            probability,
            self.threshold,
            self.label_names,
            include_report=False,
        )
        return loss_sum / max(sample_count, 1), statistics

    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        test_loader: Any,
        max_epoch: int,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        best_score = float("-inf")
        best_epoch = -1
        best_val_statistics = None
        if self.early_stopper is not None:
            self.early_stopper.reset()

        for epoch in range(1, max_epoch + 1):
            train_loss, train_statistics = self._train_epoch(train_loader, epoch)
            val_statistics = self.evaluator.evaluate(val_loader)
            score = self._monitor_value(val_statistics)
            is_best = score > best_score
            if is_best:
                best_score = score
                best_epoch = epoch
                best_val_statistics = val_statistics
                self._save_checkpoint(epoch, score)

            self.history.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_statistics=train_statistics,
                val_statistics=val_statistics,
                is_best=is_best,
            )
            logger.info(
                "Epoch %d | train loss %.4f mAP %.4f macro-F1 %.4f acc %.4f | "
                "val loss %.4f mAP %.4f macro-F1 %.4f micro-F1 %.4f "
                "acc %.4f exact-match %.4f",
                epoch,
                train_loss,
                train_statistics["mAP"],
                train_statistics["f1_macro"],
                train_statistics["hamming_accuracy"],
                val_statistics["loss"],
                val_statistics["mAP"],
                val_statistics["f1_macro"],
                val_statistics["f1_micro"],
                val_statistics["hamming_accuracy"],
                val_statistics["subset_accuracy"],
            )
            if self.early_stopper is not None and self.early_stopper.step(score):
                break

        checkpoint_path = self.ckpt_dir / "audio_best.pt"
        if not checkpoint_path.is_file():
            raise RuntimeError("Training finished without producing audio_best.pt")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        test_statistics = self.evaluator.evaluate(test_loader)
        training_time = time.perf_counter() - start_time

        timer = InferenceTimer(self.model, self.device)
        latency_ms = timer.measure_latency_per_sample(
            sample_length=self.clip_samples,
            warm_up_steps=3,
            num_steps=10,
        )
        if best_val_statistics is None:
            best_val_statistics = self.evaluator.evaluate(val_loader)
        self.history.save_summary(
            training_time=training_time,
            inference_time_ms=latency_ms,
            best_epoch=best_epoch,
            val_statistics=best_val_statistics,
            test_statistics=test_statistics,
        )
        self.history.plot_history()
        logger.info("Test report:%s", test_statistics["message"])
        logger.info("Test metrics by SNR band:%s", _format_snr_table(test_statistics["snr_metrics"]))
        return {
            "best_epoch": best_epoch,
            "best_validation": best_val_statistics,
            "test": test_statistics,
            "checkpoint_path": str(checkpoint_path),
        }
