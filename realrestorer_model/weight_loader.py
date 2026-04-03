"""
Weight loading utilities for RealRestorer.
Loads from the HuggingFace bundle layout (transformer/, vae/, text_encoder/).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file


def _list_safetensors(directory: str) -> List[str]:
    """List all .safetensors files in a directory, sorted."""
    pattern = os.path.join(directory, "*.safetensors")
    return sorted(glob.glob(pattern))


def load_safetensors_directory(directory: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load all safetensors shards from a directory and merge into one state dict."""
    files = _list_safetensors(directory)
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {directory}")
    merged = {}
    for f in files:
        merged.update(load_file(f, device=device))
    return merged


def load_transformer_weights(
    model,
    transformer_dir: str,
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Load transformer weights from the HF bundle's transformer/ directory.
    The HF bundle keys use 'inner_model.' prefix which maps directly to
    our Step1XEdit module stored at model.inner_model.
    """
    state_dict = load_safetensors_directory(transformer_dir)

    # The HF bundle stores keys as 'inner_model.xxx' -- strip that prefix
    # so they load into our Step1XEdit directly.
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("inner_model."):
            cleaned[k[len("inner_model."):]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=strict, assign=True)
    return missing, unexpected


def load_vae_weights(
    model,
    vae_dir: str,
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Load VAE weights from the HF bundle's vae/ directory.
    The HF bundle stores keys as 'inner_model.xxx'.
    """
    state_dict = load_safetensors_directory(vae_dir)

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("inner_model."):
            cleaned[k[len("inner_model."):]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=strict, assign=True)
    return missing, unexpected


def detect_transformer_config(transformer_dir: str) -> dict:
    """
    Read config.json from the transformer directory to detect model variant.
    Returns relevant flags.
    """
    config_path = os.path.join(transformer_dir, "config.json")
    config = {}
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    return {
        "version": config.get("version", "v1.1"),
        "guidance_embeds": config.get("guidance_embeds", False),
        "use_mask_token": config.get("use_mask_token", False),
        "mode": config.get("mode", "torch"),
        "in_channels": config.get("in_channels", 64),
        "out_channels": config.get("out_channels", 64),
        "hidden_size": config.get("hidden_size", 3072),
        "num_heads": config.get("num_heads", 24),
        "depth": config.get("depth", 19),
        "depth_single_blocks": config.get("depth_single_blocks", 38),
        "mlp_ratio": config.get("mlp_ratio", 4.0),
        "axes_dims_rope": list(config.get("axes_dims_rope", [16, 56, 56])),
        "theta": config.get("theta", 10000),
        "qkv_bias": config.get("qkv_bias", True),
        "vec_in_dim": config.get("vec_in_dim", 768),
        "context_in_dim": config.get("context_in_dim", 4096),
    }


def validate_bundle_path(model_root: str) -> dict:
    """
    Validate that model_root contains the expected HF bundle layout.
    Returns paths for each component.
    """
    transformer_dir = os.path.join(model_root, "transformer")
    vae_dir = os.path.join(model_root, "vae")
    text_encoder_dir = os.path.join(model_root, "text_encoder")
    processor_dir = os.path.join(model_root, "processor")

    errors = []
    if not os.path.isdir(transformer_dir):
        errors.append(f"Missing transformer/ directory at {transformer_dir}")
    if not os.path.isdir(vae_dir):
        errors.append(f"Missing vae/ directory at {vae_dir}")
    if not os.path.isdir(text_encoder_dir):
        errors.append(f"Missing text_encoder/ directory at {text_encoder_dir}")

    if errors:
        raise FileNotFoundError(
            "Invalid RealRestorer bundle layout:\n" + "\n".join(errors) + "\n"
            "Expected: ComfyUI/models/RealRestorer/<bundle>/ with "
            "transformer/, vae/, text_encoder/ subdirectories.\n"
            "Download from: https://huggingface.co/RealRestorer/RealRestorer"
        )

    return {
        "transformer_dir": transformer_dir,
        "vae_dir": vae_dir,
        "text_encoder_dir": text_encoder_dir,
        "processor_dir": processor_dir if os.path.isdir(processor_dir) else text_encoder_dir,
    }
