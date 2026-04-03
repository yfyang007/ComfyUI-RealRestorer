"""
FlowMatch Euler Discrete scheduler for RealRestorer.
Standalone -- no diffusers imports.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Optional, Sequence

import torch


def _time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def _get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15,
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        mu = _get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = _time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


class RealRestorerFlowMatchScheduler:
    """Minimal flow-matching Euler scheduler."""

    def __init__(
        self,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        shift: bool = True,
    ):
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.shift = shift
        self.timesteps = torch.tensor([], dtype=torch.float32)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[torch.device | str] = None,
        image_seq_len: Optional[int] = None,
    ) -> None:
        if image_seq_len is None:
            raise ValueError("image_seq_len is required for RealRestorerFlowMatchScheduler.")
        values = get_schedule(
            num_inference_steps,
            image_seq_len,
            base_shift=self.base_shift,
            max_shift=self.max_shift,
            shift=self.shift,
        )
        self.timesteps = torch.tensor(values, device=device, dtype=torch.float32)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        if self.timesteps.numel() == 0:
            raise RuntimeError("Call set_timesteps before step.")
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(float(timestep), device=sample.device, dtype=sample.dtype)
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)

        timestep_values = self.timesteps.to(device=sample.device, dtype=sample.dtype)
        index = int(torch.argmin(torch.abs(timestep_values - timestep.reshape(()))).item())
        next_index = min(index + 1, timestep_values.numel() - 1)
        prev_timestep = timestep_values[next_index]
        prev_sample = sample + (prev_timestep - timestep) * model_output
        return prev_sample
