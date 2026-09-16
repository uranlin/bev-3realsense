"""Depth Anything V2 Metric model wrapper."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from .. import config


class DepthEstimator:
    """Wraps Depth Anything V2 for metric monocular depth estimation."""

    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(config.DEPTH_MODEL)
        self.model = AutoModelForDepthEstimation.from_pretrained(config.DEPTH_MODEL)
        self.model = self.model.to(config.DEVICE).eval()
        self.n_params = sum(p.numel() for p in self.model.parameters()) / 1e6

    @torch.no_grad()
    def estimate(self, pil_img: Image.Image):
        """
        Run depth estimation.
        Returns:
            metric_depth: (H, W) float32 in metres
            closeness:    (H, W) float32 [0=far, 1=near] for colormap
        """
        inputs = self.processor(images=pil_img, return_tensors="pt").to(config.DEVICE)
        outputs = self.model(**inputs)
        raw = outputs.predicted_depth  # (1, h, w)

        metric_depth = (
            F.interpolate(
                raw.unsqueeze(1),
                size=(config.IMG_H, config.IMG_W),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        metric_depth = np.clip(metric_depth, config.DEPTH_MIN, config.DEPTH_MAX)

        closeness = 1.0 - (metric_depth - config.DEPTH_MIN) / (
            config.DEPTH_MAX - config.DEPTH_MIN + 1e-8
        )
        return metric_depth, closeness

    def warmup(self):
        """Run a dummy inference to compile CUDA kernels."""
        dummy = Image.new("RGB", (config.IMG_W, config.IMG_H), (128, 128, 128))
        self.estimate(dummy)
