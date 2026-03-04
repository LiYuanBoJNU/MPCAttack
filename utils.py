"""Shared utilities for adversarial attack and text generation models."""

import os
import json
import sys

import yaml
import hashlib
import base64
from typing import Dict, Any, List, Union
from omegaconf import OmegaConf
# import wandb
from config_schema import MainConfig
import torch
import torch.nn.functional as F



def info_nce_loss(feat_adv, feat_tgt, feat_src, temperature=0.2, mode='img', omega=2.0):
    """
    InfoNCE loss for pulling adv->tgt close, pushing adv->src away.

    Args:
        feat_adv: Tensor [B, D] adversarial features
        feat_tgt: Tensor [B, D] target features (positive)
        feat_src: Tensor [B, D] source features (negative)
        temperature: float, softmax temperature

    Returns:
        loss: scalar tensor
    """
    # L2 normalize
    feat_adv = F.normalize(feat_adv, dim=-1)
    feat_tgt = F.normalize(feat_tgt, dim=-1)
    feat_src = F.normalize(feat_src, dim=-1)
    # Cosine similarities
    if mode == 'text':
        sim_pos = torch.sum(feat_adv * feat_tgt, dim=-1).mean(dim=1) * omega / temperature  # [B]
        sim_neg = torch.sum(feat_adv * feat_src, dim=-1).mean(dim=1) / temperature  # [B]
    else:
        sim_pos = torch.sum(feat_adv * feat_tgt, dim=-1) * omega / temperature  # [B]
        sim_neg = torch.sum(feat_adv * feat_src, dim=-1) / temperature  # [B]
    # Stack positive and negative: shape [B, 2]
    logits = torch.stack([sim_pos, sim_neg], dim=1)
    labels = torch.zeros(feat_adv.size(0), dtype=torch.long, device=feat_adv.device)  # positives index=0
    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    return - loss


def load_api_keys() -> Dict[str, str]:
    """Load API keys from the api_keys file.
    
    Returns:
        Dict[str, str]: Dictionary containing API keys for different models
        
    Raises:
        FileNotFoundError: If no api_keys file is found
    """
    for ext in ['yaml', 'yml', 'json']:
        file_path = f'api_keys.{ext}'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                if ext in ['yaml', 'yml']:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
    
    raise FileNotFoundError(
        "API keys file not found. Please create api_keys.yaml, api_keys.yml, or api_keys.json "
        "in the root directory with your API keys."
    )


def get_api_key(model_name: str) -> str:
    """Get API key for specified model.
    
    Args:
        model_name: Name of the model to get API key for
        
    Returns:
        str: API key for the specified model
        
    Raises:
        KeyError: If API key for model is not found
    """
    api_keys = load_api_keys()
    if model_name not in api_keys:
        raise KeyError(
            f"API key for {model_name} not found in api_keys file. "
            f"Available models: {list(api_keys.keys())}"
        )
    return api_keys[model_name]


def hash_training_config(cfg: MainConfig) -> str:
    """Create a deterministic hash of training-relevant config parameters.
    
    Args:
        cfg: Configuration object containing model settings
        
    Returns:
        str: MD5 hash of the config parameters
    """
    # Convert backbone list to plain Python list
    if isinstance(cfg.model.backbone, (list, tuple)):
        backbone = list(cfg.model.backbone)
    else:
        backbone = OmegaConf.to_container(cfg.model.backbone)
        
    # Create config dict with converted values
    train_config = {
        "data": {
            "batch_size": int(cfg.data.batch_size),
            "num_samples": int(cfg.data.num_samples),
            "cle_data_path": str(cfg.data.cle_data_path),
            "tgt_data_path": str(cfg.data.tgt_data_path),
        },
        "optim": {
            "alpha": float(cfg.optim.alpha),
            "epsilon": int(cfg.optim.epsilon),
            "steps": int(cfg.optim.steps),
        },
        "model": {
            "input_res": int(cfg.model.input_res),
            "use_source_crop": bool(cfg.model.use_source_crop),
            "use_target_crop": bool(cfg.model.use_target_crop),
            "crop_scale": tuple(float(x) for x in cfg.model.crop_scale),
            "ensemble": bool(cfg.model.ensemble),
            "backbone": backbone,
        },
        "attack": cfg.attack,
    }
    
    # Convert to JSON string with sorted keys
    json_str = json.dumps(train_config, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()


# def setup_wandb(cfg: MainConfig, tags=None) -> None:
#     """Initialize Weights & Biases logging.
#
#     Args:
#         cfg: Configuration object containing wandb settings
#     """
#     config_dict = OmegaConf.to_container(cfg, resolve=True)
#     wandb.init(
#         project=cfg.wandb.project,
#         config=config_dict,
#         tags=tags,
#     )


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure exists
    """
    os.makedirs(path, exist_ok=True)


def get_output_paths(cfg: MainConfig, config_hash: str) -> Dict[str, str]:
    """Get dictionary of output paths based on config.
    
    Args:
        cfg: Configuration object
        config_hash: Hash of training config
        
    Returns:
        Dict[str, str]: Dictionary containing output paths
    """
    return {
        'output_dir': os.path.join(cfg.data.output, "img", config_hash),
        'desc_output_dir': os.path.join(cfg.data.output, "description", config_hash)
    } 