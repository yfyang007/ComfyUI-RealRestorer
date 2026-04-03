"""
RealRestorer inference pipeline.
Standalone orchestrator -- no diffusers DiffusionPipeline dependency.
Replicates the exact logic from pipeline_realrestorer.py.
"""
from __future__ import annotations

import contextlib
import math
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image

from .scheduler import RealRestorerFlowMatchScheduler


# The Qwen2.5-VL prompt prefix used by the original pipeline.
QWEN25VL_PREFIX = (
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed '
    "visual descriptions suitable for image generation. Evaluate the level of "
    "detail in the user prompt:\n"
    "- If the prompt is simple, focus on adding specifics about colors, shapes, "
    "sizes, textures, and spatial relationships to create vivid and concrete scenes.\n"
    "- If the prompt is already detailed, refine and enhance the existing details "
    "slightly without overcomplicating.\n\n"
    "Here are examples of how to transform or refine prompts:\n"
    '- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled '
    "up in a round shape, sleeping peacefully on a warm sunny windowsill, "
    "surrounded by pots of blooming red flowers.\n"
    '- User Prompt: A busy city street -> Enhanced: A bustling city street scene '
    "at dusk, featuring glowing street lamps, a diverse crowd of people in "
    "colorful clothing, and a double-decker bus passing by towering glass "
    "skyscrapers.\n\n"
    "Please generate only the enhanced description for the prompt below and "
    "avoid including any additional commentary or evaluations:\n"
    "User Prompt:"
)


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def _resize_image(img: Image.Image, img_size: int = 1024):
    """Resize preserving aspect ratio to ~img_size^2 total pixels, aligned to 16px."""
    width, height = img.size
    ratio = width / height
    if width > height:
        width_new = math.ceil(math.sqrt(img_size * img_size * ratio))
        height_new = math.ceil(width_new / ratio)
    else:
        height_new = math.ceil(math.sqrt(img_size * img_size / ratio))
        width_new = math.ceil(height_new * ratio)
    height_new = max(16, height_new // 16 * 16)
    width_new = max(16, width_new // 16 * 16)
    return img.resize((width_new, height_new), Image.LANCZOS), img.size


def _prepare_qwen_image(image: Image.Image) -> Image.Image:
    """Prepare image for Qwen2.5-VL processing (resize to valid patch grid)."""
    image = image.convert("RGB")
    min_pixels = 4 * 28 * 28
    max_pixels = 16384 * 28 * 28
    width, height = image.size
    h_bar = max(28, round(height / 28) * 28)
    w_bar = max(28, round(width / 28) * 28)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(28, math.floor(height / beta / 28) * 28)
        w_bar = max(28, math.floor(width / beta / 28) * 28)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / max(height * width, 1))
        h_bar = max(28, math.ceil(height * beta / 28) * 28)
        w_bar = max(28, math.ceil(width * beta / 28) * 28)
    return image.resize((w_bar, h_bar), Image.BICUBIC)


def _split_string(s: str):
    """Split Qwen chat template output for token-level processing."""
    s = s.replace("\u2018", '"').replace("\u2019", '"').replace("\u201c", '"').replace("\u201d", '"')
    result = []
    in_quotes = False
    temp = ""
    for idx, char in enumerate(s):
        if char == '"' and idx > 155:
            temp += char
            if not in_quotes:
                result.append(temp)
                temp = ""
            in_quotes = not in_quotes
            continue
        if in_quotes:
            result.append("\u201c" + char + "\u201d")
        else:
            temp += char
    if temp:
        result.append(temp)
    return result


def _pack_latents(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


def _unpack_latents(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return rearrange(
        x, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16), w=math.ceil(width / 16), ph=2, pw=2,
    )


def _prepare_img_ids(batch_size, packed_height, packed_width, dtype, device, axis0=0.0):
    img_ids = torch.zeros(packed_height, packed_width, 3, dtype=dtype)
    img_ids[..., 0] = axis0
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(packed_height, dtype=dtype)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(packed_width, dtype=dtype)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
    return img_ids.to(device=device, dtype=dtype)


def _process_diff_norm(diff_norm: torch.Tensor, k: float) -> torch.Tensor:
    pow_result = torch.pow(diff_norm, k)
    return torch.where(
        diff_norm > 1.0, pow_result,
        torch.where(diff_norm < 1.0, torch.ones_like(diff_norm), diff_norm),
    )


def _get_qwenvl_embeds(
    text_encoder,
    processor,
    prompts: List[str],
    ref_images: List[Optional[Image.Image]],
    edit_types: List[int],
    device: torch.device,
    dtype: torch.dtype,
    max_token_length: int = 640,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prompts (and optionally reference images) using Qwen2.5-VL."""
    model_dtype = next(text_encoder.parameters()).dtype
    batch_size = len(prompts)

    hidden_size = text_encoder.config.hidden_size
    embs = torch.zeros(batch_size, max_token_length, hidden_size, dtype=dtype, device=device)
    masks = torch.zeros(batch_size, max_token_length, dtype=torch.long, device=device)

    te_device = next(text_encoder.parameters()).device

    for idx, (prompt, ref_image, edit_type) in enumerate(zip(prompts, ref_images, edit_types)):
        messages = [{"role": "user", "content": []}]
        messages[0]["content"].append({"type": "text", "text": QWEN25VL_PREFIX})
        if edit_type != 0 and ref_image is not None:
            messages[0]["content"].append({"type": "image", "image": ref_image})
        messages[0]["content"].append({"type": "text", "text": prompt})

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True,
        )

        token_list = []
        for text_each in _split_string(text):
            txt_inputs = processor(
                text=text_each, images=None, videos=None,
                padding=True, return_tensors="pt",
            )
            token_each = txt_inputs.input_ids
            if token_each[0][0] == 2073 and token_each[0][-1] == 854:
                token_each = token_each[:, 1:-1]
            token_list.append(token_each)

        new_txt_ids = torch.cat(token_list, dim=1).to(te_device)

        if edit_type != 0 and ref_image is not None:
            image_inputs = [_prepare_qwen_image(ref_image)]
            inputs = processor(
                text=[text], images=image_inputs, padding=True, return_tensors="pt",
            )
            old_inputs_ids = inputs.input_ids.to(te_device)
            idx1 = (old_inputs_ids == 151653).nonzero(as_tuple=True)[1][0]
            idx2 = (new_txt_ids == 151653).nonzero(as_tuple=True)[1][0]
            input_ids = torch.cat([old_inputs_ids[0, :idx1], new_txt_ids[0, idx2:]], dim=0).unsqueeze(0)
            attention_mask = (input_ids > 0).long().to(te_device)
            outputs = text_encoder(
                input_ids=input_ids.to(te_device),
                attention_mask=attention_mask,
                pixel_values=inputs.pixel_values.to(te_device, dtype=model_dtype),
                image_grid_thw=inputs.image_grid_thw.to(te_device),
                output_hidden_states=True,
            )
        else:
            input_ids = new_txt_ids
            attention_mask = (input_ids > 0).long().to(te_device)
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        emb = outputs.hidden_states[-1]
        valid_length = min(max_token_length, max(0, emb.shape[1] - 217))
        if valid_length > 0:
            embs[idx, :valid_length] = emb[0, 217:217 + valid_length].to(device=device, dtype=dtype)
            masks[idx, :valid_length] = 1

    return embs, masks


def encode_vae_image(vae, image: torch.Tensor) -> torch.Tensor:
    """Encode an image tensor [0,1] -> latents via the VAE."""
    vae_device = vae.device
    vae_dtype = vae.dtype
    image = image.to(device=vae_device, dtype=vae_dtype)
    with torch.autocast(device_type=vae_device.type, enabled=False) if vae_device.type in {"cuda", "cpu"} else contextlib.nullcontext():
        latents = vae.encode(image * 2 - 1)
    return latents


def decode_vae_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents -> image tensor via the VAE."""
    vae_device = vae.device
    vae_dtype = vae.dtype
    latents = latents.to(device=vae_device, dtype=vae_dtype)
    with torch.autocast(device_type=vae_device.type, enabled=False) if vae_device.type in {"cuda", "cpu"} else contextlib.nullcontext():
        decoded = vae.decode(latents)
    return decoded


def denoise_edit(
    transformer, scheduler, latents, ref_latents,
    prompt_embeds, prompt_mask, img_ids, txt_ids,
    timesteps, guidance_scale, model_guidance,
    timesteps_truncate=0.93, process_norm_power=0.4,
    step_callback=None,
) -> torch.Tensor:
    """Edit-mode denoise loop (image restoration)."""
    total_steps = len(timesteps) - 1
    for step_idx, t in enumerate(timesteps[:-1]):
        latent_model_input = latents.repeat(2, 1, 1) if guidance_scale != -1 else latents
        ref_model_input = ref_latents.repeat(latent_model_input.shape[0], 1, 1)
        model_input = torch.cat([latent_model_input, ref_model_input], dim=1)
        t_vec = torch.full(
            (model_input.shape[0],), float(t),
            dtype=model_input.dtype, device=model_input.device,
        )
        guidance_vec = torch.full(
            (model_input.shape[0],), model_guidance,
            dtype=model_input.dtype, device=model_input.device,
        )

        pred_full = transformer(
            img=model_input,
            img_ids=img_ids,
            txt_ids=txt_ids,
            timesteps=t_vec,
            llm_embedding=prompt_embeds,
            t_vec=t_vec,
            mask=prompt_mask,
            guidance=guidance_vec,
        )

        pred = pred_full[:, :latents.shape[1]]
        if guidance_scale != -1:
            cond, uncond = pred.chunk(2, dim=0)
            if float(t) > timesteps_truncate:
                diff = cond - uncond
                diff_norm = torch.norm(diff, dim=2, keepdim=True)
                pred = uncond + guidance_scale * (cond - uncond) / _process_diff_norm(
                    diff_norm, k=process_norm_power
                )
            else:
                pred = uncond + guidance_scale * (cond - uncond)

        latents = scheduler.step(pred, t, latents)

        if step_callback is not None:
            step_callback(step_idx + 1, total_steps, latents)

    return latents


def latent_preview_image(latents: torch.Tensor, height: int, width: int) -> Image.Image:
    """
    Generate a rough RGB preview from packed latents.
    Takes channels 0, 1, 2 of the unpacked latent as an approximate RGB proxy.
    This is a fast approximation -- not a full VAE decode.
    """
    unpacked = _unpack_latents(latents[:1].float(), height, width)
    # Take first 3 of 16 channels as rough RGB
    rgb = unpacked[0, :3]
    # Normalize to 0-1 range
    for c in range(3):
        ch = rgb[c]
        ch_min = ch.min()
        ch_max = ch.max()
        if ch_max - ch_min > 1e-6:
            rgb[c] = (ch - ch_min) / (ch_max - ch_min)
        else:
            rgb[c] = torch.zeros_like(ch)
    arr = (rgb.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _offload_to_cpu(module, label=""):
    """Move a module to CPU and clear CUDA cache."""
    module.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if label:
        print(f"[RealRestorer] Offloaded {label} to CPU.", flush=True)


@torch.inference_mode()
def run_realrestorer(
    text_encoder,
    processor,
    transformer,
    vae,
    image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 28,
    guidance_scale: float = 3.0,
    seed: int = 42,
    size_level: int = 1024,
    model_guidance: float = 3.5,
    max_token_length: int = 640,
    version: str = "v1.1",
    timesteps_truncate: float = 0.93,
    process_norm_power: float = 0.4,
    device: torch.device = None,
    offload: bool = False,
    step_callback=None,
) -> Image.Image:
    """
    Full RealRestorer inference pipeline.
    Returns a single restored PIL image.

    When offload=True, each component is moved to GPU only when needed,
    then back to CPU. This trades speed for VRAM.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))

    # Prepare input image
    pil_image = image.convert("RGB")
    resized_image, original_size = _resize_image(pil_image, img_size=size_level)
    ref_images_raw = _pil_to_tensor(resized_image).unsqueeze(0).to(device=device, dtype=torch.float32)
    height, width = ref_images_raw.shape[-2:]

    # --- Phase 1: VAE encode ---
    print("[RealRestorer] Encoding reference image with VAE...", flush=True)
    if offload:
        vae.to(device)
    ref_latents_tensor = encode_vae_image(vae, ref_images_raw)
    ref_latents = _pack_latents(ref_latents_tensor.to(device=device, dtype=torch.bfloat16))
    if offload:
        _offload_to_cpu(vae, "VAE (encode)")

    # Noise
    latent_channels = getattr(vae, "latent_channels", 16)
    noise = torch.randn(
        1, latent_channels, height // 8, width // 8,
        generator=generator, device=device, dtype=torch.bfloat16,
    )
    latents = _pack_latents(noise)

    # --- Phase 2: Prompt encoding ---
    print("[RealRestorer] Encoding prompt with Qwen2.5-VL...", flush=True)
    if offload:
        text_encoder.to(device)
    pil_for_encode = _tensor_to_pil(ref_images_raw[0])
    prompt_embeds, prompt_mask = _get_qwenvl_embeds(
        text_encoder, processor,
        prompts=[prompt, negative_prompt],
        ref_images=[pil_for_encode, pil_for_encode],
        edit_types=[1, 1],
        device=device,
        dtype=next(text_encoder.parameters()).dtype,
        max_token_length=max_token_length,
    )
    if offload:
        _offload_to_cpu(text_encoder, "text_encoder")

    # Position IDs
    txt_ids = torch.zeros(
        prompt_embeds.shape[0], prompt_embeds.shape[1], 3,
        dtype=prompt_embeds.dtype, device=device,
    )
    packed_h = math.ceil(height / 16)
    packed_w = math.ceil(width / 16)
    img_ids = _prepare_img_ids(
        batch_size=prompt_embeds.shape[0],
        packed_height=packed_h, packed_width=packed_w,
        dtype=prompt_embeds.dtype, device=device, axis0=0.0,
    )
    ref_axis = 0.0 if version == "v1.0" else 1.0
    ref_img_ids = _prepare_img_ids(
        batch_size=prompt_embeds.shape[0],
        packed_height=packed_h, packed_width=packed_w,
        dtype=prompt_embeds.dtype, device=device, axis0=ref_axis,
    )
    combined_img_ids = torch.cat([img_ids, ref_img_ids], dim=1)

    # Scheduler
    scheduler = RealRestorerFlowMatchScheduler()
    scheduler.set_timesteps(
        num_inference_steps=num_inference_steps,
        device=device,
        image_seq_len=latents.shape[1],
    )
    timesteps = scheduler.timesteps.tolist()

    # --- Phase 3: Denoise ---
    print(f"[RealRestorer] Denoising ({num_inference_steps} steps)...", flush=True)
    if offload:
        transformer.to(device)

    # Wrap the external callback with height/width context
    import sys
    def _internal_step_cb(step_num, total, current_latents):
        sys.stdout.write(f"\r[RealRestorer] Step {step_num}/{total}   ")
        sys.stdout.flush()
        if step_callback is not None:
            step_callback(step_num, total, current_latents, height, width)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext():
        latents = denoise_edit(
            transformer=transformer,
            scheduler=scheduler,
            latents=latents,
            ref_latents=ref_latents,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            img_ids=combined_img_ids,
            txt_ids=txt_ids,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            model_guidance=model_guidance,
            timesteps_truncate=timesteps_truncate,
            process_norm_power=process_norm_power,
            step_callback=_internal_step_cb,
        )
    print("", flush=True)  # newline after the step counter
    if offload:
        _offload_to_cpu(transformer, "transformer")

    # --- Phase 4: VAE decode ---
    print("[RealRestorer] Decoding with VAE...", flush=True)
    if offload:
        vae.to(device)
    decoded = decode_vae_latents(vae, _unpack_latents(latents.float(), height, width))
    if offload:
        _offload_to_cpu(vae, "VAE (decode)")
    decoded = decoded.clamp(-1, 1).mul(0.5).add(0.5)
    result_pil = _tensor_to_pil(decoded[0].float())
    result_pil = result_pil.resize(original_size)

    print("[RealRestorer] Done.", flush=True)
    return result_pil
