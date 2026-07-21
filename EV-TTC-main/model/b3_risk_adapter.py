"""Fixed 3x3 local-risk modulation for the 12-channel B1-Full representation."""

from __future__ import annotations

import math

import torch
from torch import nn


class RiskModulator3x3(nn.Module):
    """Parameter-free local TTC-risk to six-scale gain mapping."""

    def __init__(self, min_activity_pixels: int = 32, inverse_ttc_floor: float = 0.1):
        super().__init__()
        self.min_activity_pixels = int(min_activity_pixels)
        self.inverse_ttc_floor = float(inverse_ttc_floor)
        self.register_buffer("low_gain", torch.tensor([0.5, 0.7, 0.9, 1.1, 1.3, 1.5]))
        self.register_buffer("mid_gain", torch.tensor([0.8, 0.9, 1.1, 1.2, 1.1, 0.9]))
        self.register_buffer("high_gain", torch.tensor([1.5, 1.3, 1.1, 0.9, 0.7, 0.5]))

    def _risk_to_gains(self, risk: torch.Tensor, enough_activity: torch.Tensor) -> torch.Tensor:
        """Map [B,3,3] inverse-TTC risk to [B,6,3,3] bounded fixed gains."""
        low = self.low_gain.view(1, 6, 1, 1)
        mid = self.mid_gain.view(1, 6, 1, 1)
        high = self.high_gain.view(1, 6, 1, 1)
        u = risk[:, None]
        low_to_mid = ((u - 0.5) / 0.5).clamp(0.0, 1.0)
        mid_to_high = ((u - 1.0) / 1.0).clamp(0.0, 1.0)
        gains = torch.where(u <= 0.5, low, low * (1.0 - low_to_mid) + mid * low_to_mid)
        gains = torch.where(u > 1.0, mid * (1.0 - mid_to_high) + high * mid_to_high, gains)
        gains = torch.where(u > 2.0, high, gains)
        neutral = torch.ones_like(gains)
        gains = torch.where(enough_activity[:, None], gains, neutral)
        return gains.clamp(0.5, 1.5)

    def compute_gains(
        self,
        previous_ttc: torch.Tensor | None,
        previous_x: torch.Tensor | None,
        reference_x: torch.Tensor | None = None,
    ):
        """Return gains, 3x3 inverse-TTC risk, and observable-pixel counts."""
        if previous_x is not None:
            batch_size, _, height, width = previous_x.shape
            device, dtype = previous_x.device, previous_x.dtype
        elif previous_ttc is not None:
            batch_size, _, height, width = previous_ttc.shape
            device, dtype = previous_ttc.device, previous_ttc.dtype
        elif reference_x is not None:
            batch_size, _, height, width = reference_x.shape
            device, dtype = reference_x.device, reference_x.dtype
        else:
            raise ValueError("at least one previous tensor is required for neutral-gain shape inference")
        if height % 3 or width % 3:
            raise ValueError(f"B3 requires image dimensions divisible by 3, got {height}x{width}")
        neutral_gains = torch.ones((batch_size, 6, 3, 3), device=device, dtype=dtype)
        neutral_risk = torch.zeros((batch_size, 3, 3), device=device, dtype=dtype)
        neutral_counts = torch.zeros((batch_size, 3, 3), device=device, dtype=torch.int64)
        if previous_ttc is None or previous_x is None:
            return neutral_gains, neutral_risk, neutral_counts
        if previous_x.shape[1] != 12:
            raise ValueError(f"B3 risk modulation requires 12 B1-Full channels, got {previous_x.shape[1]}")

        previous_ttc = previous_ttc.detach()
        previous_x = previous_x.detach()
        activity = (previous_x[:, 5] - previous_x[:, 11]).abs() > 1e-3
        ttc = previous_ttc[:, 0]
        observable = activity & torch.isfinite(ttc) & (ttc > 0)
        inverse_ttc = 1.0 / ttc.clamp_min(self.inverse_ttc_floor)
        risk = torch.zeros((batch_size, 3, 3), device=device, dtype=dtype)
        counts = torch.zeros((batch_size, 3, 3), device=device, dtype=torch.int64)
        h_step, w_step = height // 3, width // 3
        for row in range(3):
            for col in range(3):
                ys = slice(row * h_step, (row + 1) * h_step)
                xs = slice(col * w_step, (col + 1) * w_step)
                local_observable = observable[:, ys, xs].reshape(batch_size, -1)
                local_inverse = inverse_ttc[:, ys, xs].reshape(batch_size, -1)
                for batch in range(batch_size):
                    values = local_inverse[batch][local_observable[batch]]
                    counts[batch, row, col] = values.numel()
                    if values.numel() >= self.min_activity_pixels:
                        k = max(1, int(math.ceil(values.numel() * 0.10)))
                        risk[batch, row, col] = values.topk(k, largest=True).values.mean()
        gains = self._risk_to_gains(risk, counts >= self.min_activity_pixels)
        return gains, risk, counts

    @staticmethod
    def apply_gains(current_x: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
        """Broadcast [B,6,3,3] gains over the matching pos/neg 120x120 regions."""
        batch, channels, height, width = current_x.shape
        if channels != 12 or height % 3 or width % 3:
            raise ValueError(f"expected [B,12,H,W] with H/W divisible by 3, got {tuple(current_x.shape)}")
        shaped = current_x.reshape(batch, 2, 6, 3, height // 3, 3, width // 3)
        region_gains = gains[:, None, :, :, None, :, None]
        return (shaped * region_gains).reshape_as(current_x)


class B3RiskAdaptiveEVSlim(nn.Module):
    """B1-Full EV-Slim core preceded by fixed local-risk input modulation."""

    def __init__(self, base_model: nn.Module, enabled: bool = True):
        super().__init__()
        self.base_model = base_model
        self.enabled = bool(enabled)
        self.modulator = RiskModulator3x3()

    def prepare_input(self, current_x, previous_ttc=None, previous_x=None):
        if not self.enabled:
            batch = current_x.shape[0]
            gains = torch.ones((batch, 6, 3, 3), device=current_x.device, dtype=current_x.dtype)
            risk = torch.zeros((batch, 3, 3), device=current_x.device, dtype=current_x.dtype)
            counts = torch.zeros((batch, 3, 3), device=current_x.device, dtype=torch.int64)
            return current_x, {"gains": gains, "risk": risk, "activity_count": counts}
        gains, risk, counts = self.modulator.compute_gains(previous_ttc, previous_x, reference_x=current_x)
        modulated_x = self.modulator.apply_gains(current_x, gains)
        return modulated_x, {"gains": gains, "risk": risk, "activity_count": counts}

    def forward(self, current_x, previous_ttc=None, previous_x=None, return_aux: bool = False):
        modulated_x, aux = self.prepare_input(current_x, previous_ttc, previous_x)
        prediction = self.base_model(modulated_x)
        return (prediction, aux) if return_aux else prediction
