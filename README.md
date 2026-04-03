# ComfyUI-RealRestorer

Standalone ComfyUI node pack for **RealRestorer** image restoration.

**No vendored diffusers fork.** All model code is self-contained. Safe to install
into any existing ComfyUI environment without breaking dependencies.

This repository is a ComfyUI integration built on top of the official
**RealRestorer** release.

- Project page: https://yfyang007.github.io/RealRestorer/
- Paper: https://arxiv.org/abs/2603.25502
- Official repository: https://github.com/yfyang007/RealRestorer
- Hugging Face model: https://huggingface.co/RealRestorer/RealRestorer
- Hugging Face benchmark: https://huggingface.co/datasets/RealRestorer/RealIR-Bench
- Hugging Face demo: https://huggingface.co/spaces/dericky286/RealRestorer-Demo


## Why This Exists

The official RealRestorer repo requires a patched fork of `diffusers` that will
break existing ComfyUI environments. This node pack reimplements the full pipeline
from scratch using only standard PyTorch, transformers, einops, and safetensors.

If you want the original inference code, benchmark pipeline, or paper materials,
please use the official RealRestorer repository linked above.


## Requirements

Everything is already in a standard ComfyUI venv. No `pip install` needed.


## Installation

1. Clone or copy into `ComfyUI/custom_nodes/`:

```
cd ComfyUI/custom_nodes/
git clone <this-repo> ComfyUI-RealRestorer
```

2. Download the model (~42GB) into `ComfyUI/models/RealRestorer/`:

```
cd ComfyUI/models/
mkdir -p RealRestorer
cd RealRestorer
huggingface-cli download RealRestorer/RealRestorer --local-dir .
```

Expected layout:

```
ComfyUI/models/RealRestorer/
    transformer/   (safetensors + config.json)
    vae/           (safetensors + config.json)
    text_encoder/  (Qwen2.5-VL-7B weights)
    processor/     (tokenizer files)
```

3. Restart ComfyUI. The nodes appear under the **RealRestorer** category.


## Nodes

### RealRestorer Model Loader

| Setting | Default | Description |
|---|---|---|
| model | auto-detected | Scans `models/RealRestorer/` for valid bundles |
| precision | bfloat16 | Paper default. float16 uses less VRAM |
| keep_model_loaded | true | Keep in memory between runs |

### RealRestorer Sampler

| Setting | Default | Description |
|---|---|---|
| task_preset | General Restore | Pick from 10 restoration tasks or "Custom" |
| instruction | (empty) | Only used when task_preset is "Custom" |
| seed | 42 | Paper default |
| steps | 28 | Paper default (demo range: 12-40) |
| guidance_scale | 3.0 | Paper default (demo range: 1.0-6.0) |
| size_level | 1024 | Processing resolution (see tooltip) |
| device_strategy | auto | auto / full_gpu / offload_to_cpu |


## VRAM Usage

At 1024x1024 with bfloat16:

| Strategy | Peak VRAM | Speed |
|---|---|---|
| full_gpu | ~34 GB | Fastest |
| offload_to_cpu | ~18 GB peak | Slower |

With 96GB VRAM, `full_gpu` is recommended.


## Credits

- RealRestorer official repo: https://github.com/yfyang007/RealRestorer
- RealRestorer project page: https://yfyang007.github.io/RealRestorer/
- RealRestorer paper: https://arxiv.org/abs/2603.25502
- RealRestorer Hugging Face model: https://huggingface.co/RealRestorer/RealRestorer
- RealRestorer RealIR-Bench: https://huggingface.co/datasets/RealRestorer/RealIR-Bench
- RealRestorer demo: https://huggingface.co/spaces/dericky286/RealRestorer-Demo
- Base model: Step1X-Edit by StepFun AI
- ComfyUI implementation: Built with Claude (Anthropic)


## Citation

If this ComfyUI integration is useful, please also cite the official RealRestorer paper:

```bibtex
@misc{yang2026realrestorergeneralizablerealworldimage,
      title={RealRestorer: Towards Generalizable Real-World Image Restoration with Large-Scale Image Editing Models},
      author={Yufeng Yang and Xianfang Zeng and Zhangqi Jiang and Fukun Yin and Jianzhuang Liu and Wei Cheng and jinghong lan and Shiyu Liu and Yuqi Peng and Gang YU and Shifeng Chen},
      year={2026},
      eprint={2603.25502},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.25502},
}
```
