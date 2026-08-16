import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class AudioFeaturesConfig(BaseModel):
    """Waveform and log-mel settings for the 16 kHz noise dataset."""

    sample_rate: int = Field(default=16_000, gt=0)
    clip_seconds: float = Field(default=4.0, gt=0.0)
    inference_hop_seconds: float = Field(default=2.0, gt=0.0)
    window_size: int = Field(default=512, gt=0)
    hop_size: int = Field(default=160, gt=0)
    mel_bins: int = Field(default=128, gt=0)
    fmin: int = Field(default=20, ge=0)
    fmax: int = Field(default=8_000, gt=0)
    time_drop_width: int = Field(default=64, ge=0)
    time_stripes_num: int = Field(default=2, ge=0)
    freq_drop_width: int = Field(default=8, ge=0)
    freq_stripes_num: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_audio_settings(self) -> "AudioFeaturesConfig":
        if self.fmax > self.sample_rate // 2:
            raise ValueError("audio_features.fmax must not exceed the Nyquist frequency")
        if self.inference_hop_seconds > self.clip_seconds:
            raise ValueError("inference_hop_seconds must be <= clip_seconds")
        return self


class ModelConfig(BaseModel):
    backbone: str = "Cnn14MobileV2"
    pretrained: bool = False
    classes_num: int = Field(default=21, gt=0)


class SplitterConfig(BaseModel):
    """Dataset settings; the supplied train/validation/test manifests are authoritative."""

    dataset_path: str = "/marimo/21_labels_dataset"
    signal_type: Literal["mixture", "clean", "oracle_noise"] = "mixture"
    train_directory: str = "train_single"
    validation_directory: str = "validation_single"
    test_directory: str = "test_single"
    selected_labels_file: str = "selected_labels.csv"
    use_predefined_splits: bool = True
    include_video: bool = False
    save_results: bool = False

    # Kept for compatibility with copied utilities. The manifest splits are not recomputed.
    seed: int = Field(default=2026, ge=0)
    test_sample_per_class: int = Field(default=1, gt=0)
    split_strategy: str = "predefined_manifest"
    evaluation_mode: str = "holdout"
    num_folds: int = Field(default=5, gt=1)
    fold_index: Optional[int] = None
    cv_val_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)

    # Optional controlled time-varying SNR augmentation on train samples.
    dynamic_snr_enabled: bool = True
    dynamic_snr_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    dynamic_snr_min_db: float = -5.0
    dynamic_snr_max_db: float = 20.0
    dynamic_snr_control_seconds: float = Field(default=0.5, gt=0.0)

    @model_validator(mode="after")
    def validate_dataset_settings(self) -> "SplitterConfig":
        if not self.use_predefined_splits:
            raise ValueError("21_labels_dataset must use its predefined manifest splits")
        if self.dynamic_snr_min_db >= self.dynamic_snr_max_db:
            raise ValueError("dynamic_snr_min_db must be smaller than dynamic_snr_max_db")
        return self


class TrainConfig(BaseModel):
    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    monitor: Literal["macro_f1", "mAP", "loss"] = "macro_f1"
    early_stopping: bool = True
    patience: int = Field(default=15, gt=0)
    delta: float = Field(default=0.0, ge=0.0)
    cache_audio: bool = False
    num_workers: int = Field(default=-1, ge=-1)
    pin_memory: bool = True
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    use_pos_weight: bool = True
    max_pos_weight: float = Field(default=20.0, ge=1.0)
    random_seed: int = Field(default=2026, ge=0)
    ckpt_dir: str = "checkpoint"
    profile_model: bool = False
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset_splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    audio_features: AudioFeaturesConfig = Field(default_factory=AudioFeaturesConfig)

    @classmethod
    def from_json(cls, path: str = "config/train_config.json") -> "TrainConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            config = cls(**json.load(handle))

        # Relative dataset/checkpoint paths are resolved from the project directory,
        # not from the notebook's current working directory.
        project_root = config_path.parent.parent
        dataset_path = Path(config.dataset_splitter.dataset_path).expanduser()
        if not dataset_path.is_absolute():
            config.dataset_splitter.dataset_path = str((project_root / dataset_path).resolve())
        checkpoint_path = Path(config.ckpt_dir).expanduser()
        if not checkpoint_path.is_absolute():
            config.ckpt_dir = str((project_root / checkpoint_path).resolve())
        return config
