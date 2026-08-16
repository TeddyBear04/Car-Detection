from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm


def _per_class_average_precision(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    values = []
    for class_index in range(target.shape[1]):
        y_true = target[:, class_index]
        values.append(
            float("nan")
            if np.unique(y_true).size < 2
            else average_precision_score(y_true, probability[:, class_index])
        )
    return np.asarray(values, dtype=np.float64)


def _per_class_auc(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    values = []
    for class_index in range(target.shape[1]):
        y_true = target[:, class_index]
        values.append(
            float("nan")
            if np.unique(y_true).size < 2
            else roc_auc_score(y_true, probability[:, class_index])
        )
    return np.asarray(values, dtype=np.float64)


def compute_multilabel_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    label_names: Sequence[str],
    include_report: bool = True,
) -> Dict[str, Any]:
    target = target.astype(np.int32, copy=False)
    prediction = (probability >= threshold).astype(np.int32)
    average_precision = _per_class_average_precision(target, probability)
    auc = _per_class_auc(target, probability)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        target, prediction, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        target, prediction, average="micro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        target, prediction, average="weighted", zero_division=0
    )
    report = ""
    if include_report:
        report = classification_report(
            target,
            prediction,
            target_names=list(label_names),
            digits=4,
            zero_division=0,
        )
    return {
        "average_precision": average_precision,
        "mAP": float(np.nanmean(average_precision)),
        "auc": auc,
        "macro_auc": float(np.nanmean(auc)),
        "subset_accuracy": float(accuracy_score(target, prediction)),
        "accuracy": float(accuracy_score(target, prediction)),
        "hamming_accuracy": float(1.0 - hamming_loss(target, prediction)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(f1_micro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        # Compatibility keys used by the copied logger.
        "prec_macro": float(precision_macro),
        "rec_macro": float(recall_macro),
        "prec_weighted": float(precision_weighted),
        "rec_weighted": float(recall_weighted),
        "confu_matrix": multilabel_confusion_matrix(target, prediction),
        "message": "\n" + report if report else "",
        "target": target,
        "probability": probability,
        "prediction": prediction,
    }


def aggregate_windows(
    sample_ids: Sequence[str],
    probabilities: np.ndarray,
    targets: np.ndarray,
    snr_values: np.ndarray,
    reduction: str = "mean",
) -> tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    groups: "OrderedDict[str, List[int]]" = OrderedDict()
    for index, sample_id in enumerate(sample_ids):
        groups.setdefault(sample_id, []).append(index)

    clip_probabilities = []
    clip_targets = []
    clip_snrs = []
    for indexes in groups.values():
        window_values = probabilities[indexes]
        if reduction == "max":
            clip_probabilities.append(window_values.max(axis=0))
        elif reduction == "mean":
            clip_probabilities.append(window_values.mean(axis=0))
        else:
            raise ValueError(f"Unsupported window reduction: {reduction}")
        clip_targets.append(targets[indexes[0]])
        clip_snrs.append(snr_values[indexes[0]])
    return (
        list(groups.keys()),
        np.asarray(clip_probabilities, dtype=np.float32),
        np.asarray(clip_targets, dtype=np.float32),
        np.asarray(clip_snrs, dtype=np.float32),
    )


class BaseEvaluator:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.device = next(model.parameters()).device

    def evaluate(self, data_loader: Any) -> Dict[str, Any]:
        raise NotImplementedError


class AudioEvaluator(BaseEvaluator):
    def __init__(
        self,
        model: nn.Module,
        label_names: Sequence[str],
        threshold: float = 0.5,
        loss_fn: nn.Module | None = None,
        window_reduction: str = "mean",
    ) -> None:
        super().__init__(model)
        self.label_names = list(label_names)
        self.threshold = threshold
        self.loss_fn = loss_fn
        self.window_reduction = window_reduction

    def evaluate(self, data_loader: Any) -> Dict[str, Any]:
        sample_ids: List[str] = []
        probabilities = []
        targets = []
        snr_values = []
        loss_sum = 0.0
        window_count = 0
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", unit="batch", dynamic_ncols=True):
                waveform = batch["waveform"].to(self.device, non_blocking=True)
                target = batch["target"].to(self.device, non_blocking=True)
                output = self.model(waveform)
                logits = output["clipwise_output"]
                if self.loss_fn is not None:
                    batch_size = waveform.size(0)
                    loss_sum += float(self.loss_fn(output, {"target": target}).item()) * batch_size
                    window_count += batch_size
                sample_ids.extend(list(batch["audio_name"]))
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
                targets.append(target.cpu().numpy())
                snr_values.append(batch["target_snr_db"].cpu().numpy())

        window_probability = np.concatenate(probabilities, axis=0)
        window_target = np.concatenate(targets, axis=0)
        window_snr = np.concatenate(snr_values, axis=0)
        clip_ids, clip_probability, clip_target, clip_snr = aggregate_windows(
            sample_ids,
            window_probability,
            window_target,
            window_snr,
            reduction=self.window_reduction,
        )
        statistics = compute_multilabel_metrics(
            clip_target,
            clip_probability,
            self.threshold,
            self.label_names,
        )
        statistics["loss"] = loss_sum / max(window_count, 1)
        statistics["sample_ids"] = clip_ids
        statistics["target_snr_db"] = clip_snr
        statistics["num_clips"] = len(clip_ids)
        statistics["num_windows"] = len(sample_ids)

        snr_bands = {
            "[-5,0]": (clip_snr >= -5.0) & (clip_snr <= 0.0),
            "[5,10]": (clip_snr >= 5.0) & (clip_snr <= 10.0),
            "[15,20]": (clip_snr >= 15.0) & (clip_snr <= 20.0),
        }
        statistics["snr_metrics"] = {}
        for name, mask in snr_bands.items():
            if not mask.any():
                continue
            band_metrics = compute_multilabel_metrics(
                clip_target[mask],
                clip_probability[mask],
                self.threshold,
                self.label_names,
                include_report=False,
            )
            statistics["snr_metrics"][name] = {
                "samples": int(mask.sum()),
                "mAP": band_metrics["mAP"],
                "macro_f1": band_metrics["f1_macro"],
                "micro_f1": band_metrics["f1_micro"],
            }
        return statistics
