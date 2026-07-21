"""B3-v2 conservative residual local-risk modulation for B1-Full inputs."""

from __future__ import annotations

import math

import torch
from torch import nn


class B3V2RiskModulator3x3(nn.Module):
    """Parameter-free residual 3x3 modulation with risk/activity confidence."""

    def __init__(
        self,
        beta: float = 0.5,
        rho: float = 0.0,
        confidence_mode: str = "risk",
        activity_ref: float = 0.05,
        min_activity_pixels: int = 32,
        inverse_ttc_floor: float = 0.1,
    ):
        super().__init__()
        if confidence_mode not in {"risk", "risk_activity", "risk_activity_ema"}:
            raise ValueError(f"unknown confidence_mode: {confidence_mode}")
        self.beta = float(beta)
        self.rho = float(rho)
        self.confidence_mode = confidence_mode
        self.activity_ref = max(float(activity_ref), 1e-6)
        self.min_activity_pixels = int(min_activity_pixels)
        self.inverse_ttc_floor = float(inverse_ttc_floor)
        self.register_buffer("high_gain", torch.tensor([1.20, 1.12, 1.05, 0.95, 0.88, 0.80], dtype=torch.float32))

    @property
    def uses_activity(self) -> bool:
        return self.confidence_mode in {"risk_activity", "risk_activity_ema"}

    @property
    def uses_ema(self) -> bool:
        return self.confidence_mode == "risk_activity_ema" and self.rho > 0.0

    def _neutral(self, reference_x: torch.Tensor):
        batch, _, height, width = reference_x.shape
        gains = torch.ones((batch, 6, 3, 3), device=reference_x.device, dtype=reference_x.dtype)
        risk = torch.zeros((batch, 3, 3), device=reference_x.device, dtype=reference_x.dtype)
        q = torch.zeros((batch, 3, 3), device=reference_x.device, dtype=reference_x.dtype)
        active_ratio = torch.zeros((batch, 3, 3), device=reference_x.device, dtype=reference_x.dtype)
        counts = torch.zeros((batch, 3, 3), device=reference_x.device, dtype=torch.int64)
        return gains, risk, risk, q, active_ratio, counts

    def compute_raw_risk(self, previous_ttc: torch.Tensor | None, previous_x: torch.Tensor | None, reference_x: torch.Tensor):
        if previous_ttc is None or previous_x is None:
            return self._neutral(reference_x)[1:]
        if previous_x.shape[1] != 12:
            raise ValueError(f"B3-v2 requires 12 B1-Full channels, got {previous_x.shape[1]}")
        batch, _, height, width = previous_x.shape
        if height % 3 or width % 3:
            raise ValueError(f"B3-v2 requires image dimensions divisible by 3, got {height}x{width}")
        previous_ttc = previous_ttc.detach()
        previous_x = previous_x.detach()
        activity = (previous_x[:, 5] - previous_x[:, 11]).abs() > 1e-3
        ttc = previous_ttc[:, 0]
        observable = activity & torch.isfinite(ttc) & (ttc > 0)
        inverse_ttc = 1.0 / ttc.clamp_min(self.inverse_ttc_floor)
        risk = torch.zeros((batch, 3, 3), device=previous_x.device, dtype=previous_x.dtype)
        active_ratio = torch.zeros((batch, 3, 3), device=previous_x.device, dtype=previous_x.dtype)
        counts = torch.zeros((batch, 3, 3), device=previous_x.device, dtype=torch.int64)
        h_step, w_step = height // 3, width // 3
        region_pixels = h_step * w_step
        for row in range(3):
            for col in range(3):
                ys = slice(row * h_step, (row + 1) * h_step)
                xs = slice(col * w_step, (col + 1) * w_step)
                local_observable = observable[:, ys, xs].reshape(batch, -1)
                local_activity = activity[:, ys, xs].reshape(batch, -1)
                local_inverse = inverse_ttc[:, ys, xs].reshape(batch, -1)
                active_ratio[:, row, col] = local_activity.float().mean(dim=1)
                for item in range(batch):
                    values = local_inverse[item][local_observable[item]]
                    counts[item, row, col] = values.numel()
                    if values.numel() >= self.min_activity_pixels:
                        k = max(1, int(math.ceil(values.numel() * 0.10)))
                        risk[item, row, col] = values.topk(k, largest=True).values.mean()
        q_risk = ((risk - 0.5) / 0.5).clamp(0.0, 1.0)
        q_activity = (active_ratio / self.activity_ref).clamp(0.0, 1.0)
        q = q_risk * q_activity if self.uses_activity else q_risk
        q = torch.where(counts >= self.min_activity_pixels, q, torch.zeros_like(q))
        return risk, risk, q, active_ratio, counts

    def compute_gains(
        self,
        previous_ttc: torch.Tensor | None,
        previous_x: torch.Tensor | None,
        reference_x: torch.Tensor,
        risk_ema: torch.Tensor | None = None,
    ):
        gains, raw_risk, used_risk, q, active_ratio, counts = self._neutral(reference_x)
        if previous_ttc is not None and previous_x is not None:
            raw_risk, _, q_raw, active_ratio, counts = self.compute_raw_risk(previous_ttc, previous_x, reference_x)
            used_risk = raw_risk
            if self.uses_ema:
                if risk_ema is None:
                    used_risk = raw_risk
                else:
                    used_risk = self.rho * risk_ema.detach() + (1.0 - self.rho) * raw_risk
            q_risk = ((used_risk - 0.5) / 0.5).clamp(0.0, 1.0)
            q_activity = (active_ratio / self.activity_ref).clamp(0.0, 1.0)
            q = q_risk * q_activity if self.uses_activity else q_risk
            q = torch.where(counts >= self.min_activity_pixels, q, torch.zeros_like(q))
            high = self.high_gain.to(device=reference_x.device, dtype=reference_x.dtype).view(1, 6, 1, 1)
            gains = 1.0 + (self.beta * q[:, None]) * (high - 1.0)
            gains = gains.clamp(0.8, 1.2)
        return gains, raw_risk, used_risk, q, active_ratio, counts

    @staticmethod
    def apply_gains(current_x: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = current_x.shape
        if channels != 12 or height % 3 or width % 3:
            raise ValueError(f"expected [B,12,H,W] with H/W divisible by 3, got {tuple(current_x.shape)}")
        shaped = current_x.reshape(batch, 2, 6, 3, height // 3, 3, width // 3)
        return (shaped * gains[:, None, :, :, None, :, None]).reshape_as(current_x)


class B3V2RiskAdaptiveEVSlim(nn.Module):
    """B1-Full EV-Slim core preceded by conservative B3-v2 residual modulation."""

    def __init__(
        self,
        base_model: nn.Module,
        beta: float,
        rho: float,
        confidence_mode: str,
        activity_ref: float,
    ):
        super().__init__()
        self.base_model = base_model
        self.modulator = B3V2RiskModulator3x3(beta, rho, confidence_mode, activity_ref)

    def prepare_input(self, current_x, previous_ttc=None, previous_x=None, risk_ema=None):
        gains, raw_risk, used_risk, q, active_ratio, counts = self.modulator.compute_gains(
            previous_ttc, previous_x, current_x, risk_ema
        )
        modulated_x = self.modulator.apply_gains(current_x, gains)
        aux = {
            "gains": gains,
            "risk_raw": raw_risk,
            "risk_used": used_risk,
            "q": q,
            "active_ratio": active_ratio,
            "activity_count": counts,
            "risk_ema": used_risk.detach(),
        }
        return modulated_x, aux

    def forward(self, current_x, previous_ttc=None, previous_x=None, risk_ema=None, return_aux: bool = False):
        modulated_x, aux = self.prepare_input(current_x, previous_ttc, previous_x, risk_ema)
        prediction = self.base_model(modulated_x)
        return (prediction, aux) if return_aux else prediction
