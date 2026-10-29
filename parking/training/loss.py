from __future__ import annotations

import torch
from torch import nn


def focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Focal loss of CenterNet for the heatmap.

    Args:
        logits: Predicted heatmap logits (B, C, h, w)
        target: Target heatmap with gaussian peaks (B, C, h, w)

    Returns:
        Loss normalized by the number of parking spots
    """
    pred = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    positive = target.eq(1).float()
    negative = 1 - positive

    pos_loss = torch.log(pred) * (1 - pred) ** 2 * positive
    # Cells near a center are penalized less (1 - target) ** 4
    neg_loss = torch.log(1 - pred) * pred**2 * (1 - target) ** 4 * negative

    num_spots = positive.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / num_spots


class ParkingLoss(nn.Module):
    """Combined loss of heatmap and corner offsets."""

    def __init__(self, corner_weight: float = 0.1):
        """Initializes the loss.

        Args:
            corner_weight: Weight of the corner loss relative to the heatmap loss
        """
        super().__init__()
        self.corner_weight = corner_weight

    def forward(
        self,
        output: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the loss.

        Args:
            output: Network output (heatmap logits, corner offsets)
            batch: Batch of the ParkingSpotDataset

        Returns:
            Total loss and its parts for logging
        """
        heatmap, corners = output
        heatmap_loss = focal_loss(heatmap, batch["heatmap"])

        # Corners are only trained at the centers of the parking spots
        mask = batch["mask"].unsqueeze(1)
        corner_loss = (
            nn.functional.smooth_l1_loss(corners, batch["corners"], reduction="none")
            * mask
        ).sum() / mask.sum().clamp(min=1)

        loss = heatmap_loss + self.corner_weight * corner_loss
        return loss, {"heatmap": heatmap_loss.item(), "corners": corner_loss.item()}
