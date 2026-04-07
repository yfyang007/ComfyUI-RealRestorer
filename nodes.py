"""
ComfyUI nodes for RealRestorer image restoration.
No vendored diffusers -- all model code is standalone.

Model: https://huggingface.co/RealRestorer/RealRestorer
Paper: https://arxiv.org/abs/2603.25502
"""
from __future__ import annotations

import gc
import os
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

import comfy.model_management as mm
import folder_paths

# ---------------------------------------------------------------------------
#  Model directory detection
# ---------------------------------------------------------------------------

REALRESTORER_DIR = os.path.join(folder_paths.models_dir, "RealRestorer")
os.makedirs(REALRESTORER_DIR, exist_ok=True)


def _scan_model_bundles() -> list[str]:
    """
    Scan models/RealRestorer/ for valid bundles.
    A valid bundle is a directory containing a transformer/ subdirectory.
    Returns display names for the dropdown.
    """
    if not os.path.isdir(REALRESTORER_DIR):
        return []

    bundles = []

    # Check if the RealRestorer dir itself is a bundle (files placed directly)
    if os.path.isdir(os.path.join(REALRESTORER_DIR, "transformer")):
        bundles.append("RealRestorer")

    # Check subdirectories
    try:
        for name in sorted(os.listdir(REALRESTORER_DIR)):
            full = os.path.join(REALRESTORER_DIR, name)
            if os.path.isdir(full) and os.path.isdir(os.path.join(full, "transformer")):
                bundles.append(name)
    except OSError:
        pass

    return bundles if bundles else ["(no model found)"]


def _resolve_bundle_path(bundle_name: str) -> str:
    """Convert a bundle display name back to an absolute path."""
    if bundle_name == "RealRestorer":
        # Could be either the root dir or a subfolder named RealRestorer
        root_candidate = REALRESTORER_DIR
        sub_candidate = os.path.join(REALRESTORER_DIR, "RealRestorer")
        if os.path.isdir(os.path.join(root_candidate, "transformer")):
            return root_candidate
        if os.path.isdir(sub_candidate) and os.path.isdir(os.path.join(sub_candidate, "transformer")):
            return sub_candidate
        return root_candidate
    return os.path.join(REALRESTORER_DIR, bundle_name)


# ---------------------------------------------------------------------------
#  Cached pipeline components
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}
_cache_key: Optional[str] = None


def _clear_cache():
    global _cache, _cache_key
    _cache.clear()
    _cache_key = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
#  Tensor conversion helpers
# ---------------------------------------------------------------------------

def _comfy_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE (H,W,C) float [0,1] -> PIL."""
    arr = (image_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _pil_to_comfy(pil: Image.Image) -> torch.Tensor:
    """PIL -> ComfyUI IMAGE (H,W,C) float [0,1]."""
    arr = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
#  Task presets (from the paper / official demo)
# ---------------------------------------------------------------------------

TASK_PROMPTS = {
    "General Restore": "Restore the details and keep the original composition.",
    "Deblur": "Please deblur the image and make it sharper",
    "Denoise": "Please remove noise from the image.",
    "Dehaze": "Please dehaze the image",
    "Low-light Enhancement": "Please restore this low-quality image, recovering its normal brightness and clarity.",
    "Remove Compression Artifacts": "Please restore the image clarity and artifacts.",
    "Remove Lens Flare": "Please remove the lens flare and glare from the image.",
    "Remove Moire": "Please remove the moire patterns from the image",
    "Remove Rain": "Please remove the rain from the image and restore its clarity.",
    "Remove Reflection": "Please remove the reflection from the image.",
    "Custom": "",
}


# ---------------------------------------------------------------------------
#  Node: Load RealRestorer Model
# ---------------------------------------------------------------------------

class RealRestorerModelLoader:
    """Load the RealRestorer model for image restoration."""

    DESCRIPTION = (
        "Loads a RealRestorer model bundle.\n\n"
        "Place the HuggingFace model files in:\n"
        "  ComfyUI/models/RealRestorer/\n\n"
        "Download from:\n"
        "  https://huggingface.co/RealRestorer/RealRestorer\n\n"
        "Expected layout: transformer/, vae/, text_encoder/, processor/ subdirectories."
    )

    @classmethod
    def INPUT_TYPES(cls):
        bundles = _scan_model_bundles()
        return {
            "required": {
                "model": (bundles, {
                    "default": bundles[0] if bundles else "(no model found)",
                    "tooltip": (
                        "Select the RealRestorer model to load. "
                        "Place the HuggingFace download in ComfyUI/models/RealRestorer/. "
                        "The folder should contain transformer/, vae/, and text_encoder/ subdirectories."
                    ),
                }),
                "precision": (["bfloat16", "float16"], {
                    "default": "bfloat16",
                    "tooltip": (
                        "Model weight precision. bfloat16 is recommended (paper default). "
                        "float16 uses slightly less VRAM but may reduce quality."
                    ),
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Keep the model in memory between runs. "
                        "Disable to free VRAM after each queue execution, "
                        "at the cost of reloading on the next run."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("RR_MODEL",)
    RETURN_NAMES = ("RR_Model",)
    FUNCTION = "load"
    CATEGORY = "RealRestorer"

    def load(self, model: str, precision: str, keep_model_loaded: bool):
        global _cache, _cache_key

        if model == "(no model found)":
            raise RuntimeError(
                "No RealRestorer model found.\n"
                "Download from: https://huggingface.co/RealRestorer/RealRestorer\n"
                "Place files in: ComfyUI/models/RealRestorer/"
            )

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        dtype = dtype_map[precision]

        model_root = _resolve_bundle_path(model)
        cache_key = f"{model_root}|{precision}"

        if _cache_key == cache_key and _cache:
            # Update keep_loaded preference even on cache hit
            _cache["keep_loaded"] = keep_model_loaded
            print("[RealRestorer] Using cached model.", flush=True)
            return (_cache,)

        # Clear any old cache
        if _cache:
            _clear_cache()

        # Validate bundle
        from .realrestorer_model.weight_loader import validate_bundle_path, detect_transformer_config
        paths = validate_bundle_path(model_root)

        # Load transformer config
        tr_config = detect_transformer_config(paths["transformer_dir"])

        # Build transformer
        print("[RealRestorer] Building transformer...", flush=True)
        from .realrestorer_model.dit import Step1XEdit, Step1XParams
        params = Step1XParams(
            in_channels=tr_config["in_channels"],
            out_channels=tr_config["out_channels"],
            vec_in_dim=tr_config["vec_in_dim"],
            context_in_dim=tr_config["context_in_dim"],
            hidden_size=tr_config["hidden_size"],
            mlp_ratio=tr_config["mlp_ratio"],
            num_heads=tr_config["num_heads"],
            depth=tr_config["depth"],
            depth_single_blocks=tr_config["depth_single_blocks"],
            axes_dim=tr_config["axes_dims_rope"],
            theta=tr_config["theta"],
            qkv_bias=tr_config["qkv_bias"],
            mode="torch",
            version=tr_config["version"],
            guidance_embed=tr_config["guidance_embeds"],
            use_mask_token=tr_config["use_mask_token"],
        )

        with torch.device("meta"):
            transformer = Step1XEdit(params)

        from .realrestorer_model.weight_loader import load_transformer_weights
        print("[RealRestorer] Loading transformer weights...", flush=True)
        missing, unexpected = load_transformer_weights(transformer, paths["transformer_dir"])
        if missing:
            print(f"[RealRestorer] Transformer missing keys: {len(missing)}", flush=True)
        transformer = transformer.to(dtype=dtype)
        transformer.eval()
        transformer.requires_grad_(False)

        # Build VAE
        print("[RealRestorer] Building VAE...", flush=True)
        from .realrestorer_model.components import AutoEncoder
        with torch.device("meta"):
            vae = AutoEncoder()

        from .realrestorer_model.weight_loader import load_vae_weights
        print("[RealRestorer] Loading VAE weights...", flush=True)
        load_vae_weights(vae, paths["vae_dir"])
        vae = vae.to(dtype=torch.float32)
        vae.eval()
        vae.requires_grad_(False)

        # Load text encoder (Qwen2.5-VL)
        print("[RealRestorer] Loading Qwen2.5-VL text encoder...", flush=True)
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            paths["text_encoder_dir"],
            torch_dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        text_encoder.eval()
        text_encoder.requires_grad_(False)

        processor = AutoProcessor.from_pretrained(
            paths["processor_dir"],
            min_pixels=256 * 28 * 28,
            max_pixels=324 * 28 * 28,
            local_files_only=True,
        )

        print("[RealRestorer] Model loaded successfully.", flush=True)

        _cache = {
            "transformer": transformer,
            "vae": vae,
            "text_encoder": text_encoder,
            "processor": processor,
            "version": tr_config["version"],
            "dtype": dtype,
            "keep_loaded": keep_model_loaded,
        }
        _cache_key = cache_key

        return (_cache,)


# ---------------------------------------------------------------------------
#  Node: RealRestorer Sampler
# ---------------------------------------------------------------------------

class RealRestorerSampler:
    """Run RealRestorer image restoration on an input image."""

    DESCRIPTION = (
        "Restores a degraded image using RealRestorer.\n\n"
        "Select a Task Preset for common restoration tasks, or choose\n"
        "'Custom' and write your own instruction.\n\n"
        "Paper defaults: 28 steps, guidance 3.0, size 1024, seed 42."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "RR_Model": ("RR_MODEL",),
                "image": ("IMAGE",),
                "task_preset": (list(TASK_PROMPTS.keys()), {
                    "default": "General Restore",
                    "tooltip": (
                        "Select a restoration task. Each preset uses the prompt recommended "
                        "in the RealRestorer paper for that degradation type. "
                        "Choose 'Custom' to write your own instruction below."
                    ),
                }),
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Only used when Task Preset is 'Custom'",
                    "tooltip": (
                        "Custom restoration instruction. Only used when Task Preset is set to 'Custom'. "
                        "For all other presets, the paper-recommended prompt is used automatically."
                    ),
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Random seed. Paper default: 42.",
                }),
                "steps": ("INT", {
                    "default": 28, "min": 12, "max": 100,
                    "tooltip": (
                        "Number of denoising steps. Paper default: 28. "
                        "The official demo allows 12-40. Lower = faster but less detail."
                    ),
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 6.0, "step": 0.1,
                    "tooltip": (
                        "CFG guidance scale. Paper default: 3.0. "
                        "The official demo allows 1.0-6.0. "
                        "Higher values follow the prompt more strongly."
                    ),
                }),
                "size_level": ("INT", {
                    "default": 1024, "min": 256, "max": 4096, "step": 8,
                    "tooltip": (
                        "Target resolution for processing. Paper default: 1024. "
                        "The input image is resized so its total pixel area is roughly "
                        "size_level x size_level (preserving aspect ratio), processed at "
                        "that resolution, then resized back to original dimensions. "
                        "Higher = better quality but more VRAM (~34GB at 1024)."
                    ),
                }),
                "device_strategy": (
                    ["auto", "full_gpu", "offload_to_cpu", "sequential_offload"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "How to manage GPU memory.\n\n"
                            "- auto: Uses full_gpu if >40GB VRAM, offload_to_cpu if >26GB, "
                            "otherwise sequential_offload.\n"
                            "- full_gpu: All components stay on GPU. Fastest. ~34GB at size 1024.\n"
                            "- offload_to_cpu: Components move to CPU when not in use. "
                            "Slower but ~24GB peak (the transformer weight size).\n"
                            "- sequential_offload: Pre-loads as many transformer blocks to "
                            "GPU as VRAM allows, streams the rest from CPU. On a 24GB card "
                            "most blocks stay on GPU with only ~12 streaming. Enables cards "
                            "that can't fit the full transformer."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "restore"
    CATEGORY = "RealRestorer"

    def restore(
        self,
        RR_Model: dict,
        image: torch.Tensor,
        task_preset: str,
        instruction: str,
        seed: int,
        steps: int,
        guidance_scale: float,
        size_level: int,
        device_strategy: str,
    ):
        transformer = RR_Model["transformer"]
        vae = RR_Model["vae"]
        text_encoder = RR_Model["text_encoder"]
        processor = RR_Model["processor"]
        version = RR_Model["version"]

        # Resolve prompt: presets take priority; instruction only for Custom
        if task_preset == "Custom":
            prompt = instruction.strip()
            if not prompt:
                prompt = "Restore the details and keep the original composition."
                print(
                    "[RealRestorer] Custom preset selected but instruction is empty. "
                    "Using default restoration prompt.",
                    flush=True,
                )
        else:
            prompt = TASK_PROMPTS[task_preset]

        # Device strategy
        device = mm.get_torch_device()

        if device_strategy == "auto":
            total_vram = 0
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory
            if total_vram > 40 * 1024**3:
                device_strategy = "full_gpu"
            elif total_vram > 26 * 1024**3:
                device_strategy = "offload_to_cpu"
            else:
                device_strategy = "sequential_offload"
            print(f"[RealRestorer] Auto-selected device strategy: {device_strategy}", flush=True)

        from .realrestorer_model.pipeline import run_realrestorer

        use_offload = device_strategy in ("offload_to_cpu", "sequential_offload")
        use_sequential = (device_strategy == "sequential_offload")

        if not use_offload:
            # full_gpu: move everything to GPU once
            transformer.to(device)
            vae.to(device)
            text_encoder.to(device)

        # Set up ComfyUI progress bar (step counter only)
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(steps)
        except Exception:
            pbar = None

        def _step_callback(step_num, total, current_latents, height, width):
            if pbar is not None:
                pbar.update_absolute(step_num, total)

        out_frames = []
        for b in range(image.shape[0]):
            pil_in = _comfy_to_pil(image[b])

            if image.shape[0] > 1:
                print(f"[RealRestorer] Frame {b + 1}/{image.shape[0]}", flush=True)

            result_pil = run_realrestorer(
                text_encoder=text_encoder,
                processor=processor,
                transformer=transformer,
                vae=vae,
                image=pil_in,
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=seed + b,
                size_level=size_level,
                model_guidance=3.5,
                max_token_length=640,
                version=version,
                device=device,
                offload=use_offload,
                sequential_offload=use_sequential,
                step_callback=_step_callback,
            )

            out_frames.append(_pil_to_comfy(result_pil))

        # Respect keep_model_loaded preference from the loader
        if not RR_Model.get("keep_loaded", True):
            transformer.to("cpu")
            vae.to("cpu")
            text_encoder.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        out = torch.stack(out_frames, dim=0)
        return (out,)


# ---------------------------------------------------------------------------
#  Node mappings
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "RealRestorerModelLoader": RealRestorerModelLoader,
    "RealRestorerSampler": RealRestorerSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RealRestorerModelLoader": "RealRestorer Model Loader",
    "RealRestorerSampler": "RealRestorer Sampler",
}
