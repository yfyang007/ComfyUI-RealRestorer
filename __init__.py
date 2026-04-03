"""
ComfyUI-RealRestorer
Standalone ComfyUI node pack for RealRestorer image restoration.
No vendored diffusers fork -- all model code is self-contained.

Model: https://huggingface.co/RealRestorer/RealRestorer
Paper: https://arxiv.org/abs/2603.25502
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = None
