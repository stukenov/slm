"""Flow/consistency matching for latent TTS generation."""

from __future__ import annotations

import torch
import torch.nn as nn


class FlowMatchingLoss(nn.Module):
    """Conditional flow matching loss.

    Given clean latents x1 and noise x0, we interpolate:
        x_t = (1 - t) * x0 + t * x1

    The model predicts the velocity: v = x1 - x0
    Loss is ||model(x_t, t) - v||
    """

    def __init__(self, sigma_min: float = 0.001, loss_type: str = "huber"):
        super().__init__()
        self.sigma_min = sigma_min
        self.loss_type = loss_type

    def forward(
        self,
        model_output: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            model_output: predicted velocity (B, N, D)
            x0: noise (B, N, D)
            x1: clean latents (B, N, D)
            mask: (B, N) validity mask (1 = valid, 0 = padded)
        """
        target = x1 - x0  # true velocity

        if self.loss_type == "huber":
            loss = nn.functional.huber_loss(model_output, target, reduction="none")
        else:
            loss = nn.functional.mse_loss(model_output, target, reduction="none")

        # loss: (B, N, D)
        loss = loss.mean(dim=-1)  # (B, N)

        if mask is not None:
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss


@torch.no_grad()
def sample_euler(
    model: nn.Module,
    text_tokens: torch.Tensor,
    num_frames: int,
    latent_dim: int,
    num_steps: int = 8,
    text_padding_mask: torch.Tensor | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Euler ODE sampler for flow matching.

    Integrates from noise (t=0) to clean (t=1) in num_steps.

    Returns:
        latents: (B, N, D) generated latent frames
    """
    B = text_tokens.shape[0]
    x = torch.randn(B, num_frames, latent_dim, device=device)

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)
        velocity = model(x, text_tokens, t, text_padding_mask=text_padding_mask)
        x = x + velocity * dt

    return x
