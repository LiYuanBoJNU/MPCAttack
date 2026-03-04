from dataclasses import dataclass
from typing import Optional
from hydra.core.config_store import ConfigStore



@dataclass
class WandbConfig:
    """Wandb-specific configuration"""

    entity: str = "???"  # fill your wandb entity
    project: str = "local_adversarial_attack"


@dataclass
class BlackboxConfig:
    """Configuration for blackbox model evaluation"""

    model_name: str = "gpt4v"  # Can be gpt4v, claude, gemini, gpt_score
    batch_size: int = 1
    timeout: int = 30

@dataclass
class Blackbox_open_source_Config:
    """Configuration for blackbox model evaluation"""

    model_name: str = "Qwen2.5-VL-3B-Instruct"
    batch_size: int = 1
    timeout: int = 30



@dataclass
class DataConfig:
    """Data loading configuration"""

    batch_size: int = 1
    num_samples: int = 1000
    cle_data_path: str = "resources/images/bigscale"
    tgt_data_path: str = "resources/images/target_images"
    output: str = "./Ours"


@dataclass
class OptimConfig:
    """Optimization parameters"""

    alpha: float = 1.0
    epsilon: int = 16
    steps: int = 300


@dataclass
class ModelConfig:
    """Model-specific parameters"""

    input_res: int = 336
    use_source_crop: bool = True
    use_target_crop: bool = True
    crop_scale: tuple = (0.5, 0.9)
    ensemble: bool = True
    device: str = "cuda"  # Can be "cpu", "cuda:0", "cuda:1", etc.
    backbone: list = (
        "L336",
        "B16",
        "B32",
        "Laion",
    )  # List of models to use: L336, B16, B32, Laion


@dataclass
class TrainConfig:
    """Optimization parameters"""
    batch_size: int = 1
    lr: float = 3e-2
    epochs: int = 10000
    eps: int = 16

@dataclass
class MainConfig:
    """Main configuration combining all sub-configs"""

    data: DataConfig = DataConfig()
    optim: OptimConfig = OptimConfig()
    model: ModelConfig = ModelConfig()
    wandb: WandbConfig = WandbConfig()
    blackbox: BlackboxConfig = BlackboxConfig()
    blackbox_open_source: Blackbox_open_source_Config = Blackbox_open_source_Config()
    attack: str = "fgsm"  # can be [fgsm, mifgsm, pgd]
    train: TrainConfig = TrainConfig()


# register config for different setting
@dataclass
class Ensemble3ModelsConfig(MainConfig):
    """Configuration for ensemble_3models.py"""

    data: DataConfig = DataConfig(batch_size=1)
    model: ModelConfig = ModelConfig(
        use_source_crop=True, use_target_crop=True, backbone=["B16", "B32", "Laion"]
    )


# Register configs with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)
cs.store(name="ensemble_3models", node=Ensemble3ModelsConfig)
