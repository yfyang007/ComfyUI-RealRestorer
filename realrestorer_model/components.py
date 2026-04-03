"""
Qwen2Connector -- bridges Qwen2.5-VL hidden states into the DiT.
AutoEncoder -- Flux-style VAE for RealRestorer.
"""
from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, nn

from .layers import (
    MLP,
    TextProjection,
    TimestepEmbedder,
    apply_gate,
    attention,
)


# --------------------------------------------------------------------------- #
#  RMSNorm (connector-local, uses LayerNorm-style interface)
# --------------------------------------------------------------------------- #

class ConnectorRMSNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine=True, eps: float = 1e-6,
                 device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


def _get_norm_layer(norm_layer):
    if norm_layer == "layer":
        return nn.LayerNorm
    if norm_layer == "rms":
        return ConnectorRMSNorm
    raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")


def _get_activation_layer(act_type):
    if act_type == "gelu":
        return lambda: nn.GELU()
    if act_type == "gelu_tanh":
        return lambda: nn.GELU(approximate="tanh")
    if act_type == "relu":
        return nn.ReLU
    if act_type == "silu":
        return nn.SiLU
    raise ValueError(f"Unknown activation type: {act_type}")


# --------------------------------------------------------------------------- #
#  Token refiner blocks
# --------------------------------------------------------------------------- #

class IndividualTokenRefinerBlock(nn.Module):
    def __init__(self, hidden_size, heads_num, mlp_width_ratio=4.0,
                 mlp_drop_rate=0.0, act_type="silu", qk_norm=False,
                 qk_norm_type="layer", qkv_bias=True, need_CA=False,
                 dtype=None, device=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.need_CA = need_CA
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.self_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = _get_norm_layer(qk_norm_type)
        self.self_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.self_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.self_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        act_layer = _get_activation_layer(act_type)
        self.mlp = MLP(
            in_channels=hidden_size, hidden_channels=mlp_hidden_dim,
            act_layer=act_layer, drop=mlp_drop_rate, **factory_kwargs,
        )
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs),
        )
        if self.need_CA:
            self.cross_attnblock = CrossAttnBlock(
                hidden_size=hidden_size, heads_num=heads_num,
                mlp_width_ratio=mlp_width_ratio, mlp_drop_rate=mlp_drop_rate,
                act_type=act_type, qk_norm=qk_norm, qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias, **factory_kwargs,
            )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c, attn_mask=None, y=None):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)
        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)
        attn = attention(q, k, v, mode="torch", attn_mask=attn_mask)
        x = x + apply_gate(self.self_attn_proj(attn), gate_msa)
        if self.need_CA:
            x = self.cross_attnblock(x, c, attn_mask, y)
        x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, hidden_size, heads_num, mlp_width_ratio=4.0,
                 mlp_drop_rate=0.0, act_type="silu", qk_norm=False,
                 qk_norm_type="layer", qkv_bias=True, dtype=None, device=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.norm1_2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.self_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.self_attn_kv = nn.Linear(hidden_size, hidden_size * 2, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = _get_norm_layer(qk_norm_type)
        self.self_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.self_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.self_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        act_layer = _get_activation_layer(act_type)
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c, attn_mask=None, y=None):
        gate_msa, _ = self.adaLN_modulation(c).chunk(2, dim=1)
        norm_x = self.norm1(x)
        norm_y = self.norm1_2(y)
        q = self.self_attn_q(norm_x)
        q = rearrange(q, "B L (H D) -> B L H D", H=self.heads_num)
        kv = self.self_attn_kv(norm_y)
        k, v = rearrange(kv, "B L (K H D) -> K B L H D", K=2, H=self.heads_num)
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)
        attn = attention(q, k, v, mode="torch", attn_mask=attn_mask)
        return x + apply_gate(self.self_attn_proj(attn), gate_msa)


class IndividualTokenRefiner(nn.Module):
    def __init__(self, hidden_size, heads_num, depth, mlp_width_ratio=4.0,
                 mlp_drop_rate=0.0, act_type="silu", qk_norm=False,
                 qk_norm_type="layer", qkv_bias=True, need_CA=False,
                 dtype=None, device=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.need_CA = need_CA
        self.blocks = nn.ModuleList([
            IndividualTokenRefinerBlock(
                hidden_size=hidden_size, heads_num=heads_num,
                mlp_width_ratio=mlp_width_ratio, mlp_drop_rate=mlp_drop_rate,
                act_type=act_type, qk_norm=qk_norm, qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias, need_CA=self.need_CA, **factory_kwargs,
            )
            for _ in range(depth)
        ])

    def forward(self, x, c, mask=None, y=None):
        self_attn_mask = None
        if mask is not None:
            batch_size = mask.shape[0]
            seq_len = mask.shape[1]
            mask = mask.to(x.device)
            self_attn_mask_1 = mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            self_attn_mask[:, :, :, 0] = True
        for block in self.blocks:
            x = block(x, c, self_attn_mask, y)
        return x


class SingleTokenRefiner(nn.Module):
    def __init__(self, in_channels, hidden_size, heads_num, depth,
                 mlp_width_ratio=4.0, mlp_drop_rate=0.0, act_type="silu",
                 qk_norm=False, qk_norm_type="layer", qkv_bias=True,
                 need_CA=False, dtype=None, device=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.need_CA = need_CA
        self.input_embedder = nn.Linear(in_channels, hidden_size, bias=True, **factory_kwargs)
        if self.need_CA:
            self.input_embedder_CA = nn.Linear(in_channels, hidden_size, bias=True, **factory_kwargs)
        act_layer = _get_activation_layer(act_type)
        self.t_embedder = TimestepEmbedder(hidden_size, act_layer, **factory_kwargs)
        self.c_embedder = TextProjection(in_channels, hidden_size, act_layer, **factory_kwargs)
        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size, heads_num=heads_num, depth=depth,
            mlp_width_ratio=mlp_width_ratio, mlp_drop_rate=mlp_drop_rate,
            act_type=act_type, qk_norm=qk_norm, qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias, need_CA=need_CA, **factory_kwargs,
        )

    def forward(self, x, t, mask=None, y=None):
        timestep_aware = self.t_embedder(t)
        if mask is None:
            context_aware = x.mean(dim=1)
        else:
            mask_float = mask.unsqueeze(-1)
            context_aware = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        context_aware = self.c_embedder(context_aware)
        c = timestep_aware + context_aware
        x = self.input_embedder(x)
        if self.need_CA:
            y = self.input_embedder_CA(y)
            x = self.individual_token_refiner(x, c, mask, y)
        else:
            x = self.individual_token_refiner(x, c, mask)
        return x


class Qwen2Connector(nn.Module):
    def __init__(self, in_channels=3584, hidden_size=4096, heads_num=32,
                 depth=2, need_CA=False, device=None, dtype=torch.bfloat16,
                 version="v1.0"):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.S = SingleTokenRefiner(
            in_channels=in_channels, hidden_size=hidden_size,
            heads_num=heads_num, depth=depth, need_CA=need_CA,
            **factory_kwargs,
        )
        self.global_proj_out = nn.Linear(in_channels, 768)
        self.version = version
        if self.version == "v1.0":
            self.scale_factor = nn.Parameter(torch.zeros(1))
            with torch.no_grad():
                self.scale_factor.data += -(1 - 0.09)

    def forward(self, x, t, mask):
        t = t * 1000
        mask_float = mask.unsqueeze(-1)
        x_mean = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        if self.version == "v1.0":
            x_mean = x_mean * (1 + self.scale_factor.to(x.dtype))
        global_out = self.global_proj_out(x_mean)
        encoder_hidden_states = self.S(x, t, mask)
        return encoder_hidden_states, global_out


# =========================================================================== #
#  AutoEncoder (Flux-VAE)
# =========================================================================== #

def _swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.norm(hidden_states)
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        hidden_states = nn.functional.scaled_dot_product_attention(q, k, v)
        return rearrange(hidden_states, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = _swish(self.norm1(x))
        h = self.conv1(h)
        h = _swish(self.norm2(h))
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        x = nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, resolution, in_channels, ch, ch_mult, num_res_blocks, z_channels):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        in_ch_mult = (1, *tuple(ch_mult))
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        hidden_states = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hidden_states[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hidden_states.append(h)
            if i_level != self.num_resolutions - 1:
                hidden_states.append(self.down[i_level].downsample(hidden_states[-1]))
        h = hidden_states[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = _swish(h)
        return self.conv_out(h)


class Decoder(nn.Module):
    def __init__(self, ch, out_ch, ch_mult, num_res_blocks, in_channels, resolution, z_channels):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = _swish(h)
        return self.conv_out(h)


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        return mean


class AutoEncoder(nn.Module):
    def __init__(self, resolution=256, in_channels=3, ch=128, out_ch=3,
                 ch_mult=(1, 2, 4, 4), num_res_blocks=2, z_channels=16,
                 scale_factor=0.3611, shift_factor=0.1159):
        super().__init__()
        ch_mult = list(ch_mult)
        self.encoder = Encoder(resolution=resolution, in_channels=in_channels,
                               ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               z_channels=z_channels)
        self.decoder = Decoder(resolution=resolution, in_channels=in_channels,
                               ch=ch, out_ch=out_ch, ch_mult=ch_mult,
                               num_res_blocks=num_res_blocks, z_channels=z_channels)
        self.reg = DiagonalGaussian()
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor
        self.latent_channels = z_channels

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        return self.scale_factor * (z - self.shift_factor)

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)
