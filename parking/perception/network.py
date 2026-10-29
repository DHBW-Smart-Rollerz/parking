"""Neural network for the parking spot detection.

The network follows the idea of CenterNet: For every cell of the output grid it
predicts

- a heatmap per class (free, occupied, ...), whose peaks are the centers of the
  parking spots, and
- the offsets of the four corners relative to the cell (8 values).

This allows any number of parking spots per image without anchors or
postprocessing of boxes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

STRIDE = 4  # Output grid cell size in px of the network input
NUM_CORNERS = 4


def conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    """Two 3x3 convolutions with batch norm and ReLU.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride of the first convolution (2 halves the resolution)

    Returns:
        The block as sequential module
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ParkingNet(nn.Module):
    """Small encoder-decoder network for the parking spot keypoint detection."""

    def __init__(self, num_classes: int, width: int = 16):
        """Initializes the network.

        Args:
            num_classes: Number of parking spot classes (heatmap channels)
            width: Number of channels of the first layer, the deeper layers use
                multiples of it
        """
        super().__init__()
        self.num_classes = num_classes

        # Encoder: stride 2, 4, 8, 16
        self.enc1 = conv_block(1, width, stride=2)
        self.enc2 = conv_block(width, 2 * width, stride=2)
        self.enc3 = conv_block(2 * width, 4 * width, stride=2)
        self.enc4 = conv_block(4 * width, 8 * width, stride=2)

        # Decoder with skip connections back to stride 4
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = conv_block(8 * width + 4 * width, 4 * width)
        self.dec2 = conv_block(4 * width + 2 * width, 4 * width)

        self.heatmap_head = self._head(4 * width, num_classes)
        self.corner_head = self._head(4 * width, 2 * NUM_CORNERS)

        # Start with a low probability for parking spot centers (CenterNet trick),
        # otherwise the heatmap loss is dominated by the background at the start
        self.heatmap_head[-1].bias.data.fill_(-2.19)

    @staticmethod
    def _head(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Grayscale images with shape (B, 1, H, W) and values in [0, 1].
                H and W have to be divisible by 16.

        Returns:
            Heatmap logits (B, num_classes, H/4, W/4) and corner offsets
            (B, 8, H/4, W/4) in output cells
        """
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))

        return self.heatmap_head(d2), self.corner_head(d2)


def decode(
    heatmap: torch.Tensor,
    corners: torch.Tensor,
    score_threshold: float = 0.3,
    max_spots: int = 20,
) -> list[tuple[int, float, np.ndarray]]:
    """Converts the network output of one image into parking spots.

    Args:
        heatmap: Heatmap logits with shape (num_classes, h, w)
        corners: Corner offsets with shape (8, h, w)
        score_threshold: Minimum score of a parking spot
        max_spots: Maximum number of returned parking spots

    Returns:
        List of (class id, score, corners in px of the network input with shape
        (4, 2)), sorted by score
    """
    scores = torch.sigmoid(heatmap)

    # Non-maximum suppression: only keep local maxima in a 3x3 neighbourhood
    peaks = scores == nn.functional.max_pool2d(scores, 3, stride=1, padding=1)
    scores = scores * peaks

    _, h, w = scores.shape
    top_scores, top_indices = scores.flatten().topk(min(max_spots, scores.numel()))

    spots = []
    for score, index in zip(top_scores.tolist(), top_indices.tolist(), strict=True):
        if score < score_threshold:
            break
        class_id, cell = divmod(index, h * w)
        y, x = divmod(cell, w)
        offsets = corners[:, y, x].reshape(NUM_CORNERS, 2).cpu().numpy()
        spot_corners = (offsets + np.array([x, y])) * STRIDE
        spots.append((class_id, score, spot_corners))
    return spots
