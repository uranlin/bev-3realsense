"""Image colormaps and encoding utilities."""

import base64
import io

import numpy as np
from PIL import Image


def turbo_colormap(values: np.ndarray) -> np.ndarray:
    """Apply turbo-like colormap to [0,1] values → (H, W, 3) uint8."""
    t = np.clip(values, 0, 1).astype(np.float32)
    r = np.clip(np.where(t < 0.5, 2 * t * 2.5, 1 - (t - 0.5) * 3), 0, 1)
    g = np.clip(
        np.where(t < 0.35, t * 4, np.where(t < 0.65, 1.0, 1 - (t - 0.65) * 3)), 0, 1
    )
    b = np.clip(
        np.where(
            t < 0.3, 0.5 + t * 2, np.where(t < 0.6, 1 - (t - 0.3) * 2, (t - 0.6) * 2.5)
        ),
        0,
        1,
    )
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def encode_image(arr: np.ndarray, display_size: tuple = None) -> str:
    """Encode (H,W,3) uint8 array → base64 PNG data URI."""
    img = Image.fromarray(arr)
    if display_size:
        img = img.resize(display_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
