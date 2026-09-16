"""Camera intrinsic / extrinsic helpers."""

import numpy as np

from .. import config


def make_intrinsic(fx=None, fy=None, cx=None, cy=None):
    """Build 3×3 pinhole intrinsic matrix, defaulting to scaled RealSense values."""
    w, h = config.IMG_W, config.IMG_H
    sx = w / 640.0
    sy = h / 480.0
    K = np.array(
        [
            [
                config.REALSENSE_FX * sx if fx is None else fx,
                0,
                config.REALSENSE_CX * sx if cx is None else cx,
            ],
            [
                0,
                config.REALSENSE_FY * sy if fy is None else fy,
                config.REALSENSE_CY * sy if cy is None else cy,
            ],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Intrinsics require finite values and positive focal lengths")
    return K


def scale_intrinsic(K, source_size, target_size):
    """Scale a pinhole matrix for an uncropped resize; sizes are (width, height)."""
    sw, sh = source_size
    tw, th = target_size
    if min(sw, sh, tw, th) <= 0:
        raise ValueError("Image dimensions must be positive")
    scaled = np.array(K, dtype=np.float64, copy=True)
    scaled[0, :] *= tw / sw
    scaled[1, :] *= th / sh
    return scaled


def make_extrinsic(height=1.5, pitch_deg=-5.0, roll_deg=0.0, yaw_deg=0.0):
    """
    Build 4×4 camera-to-world extrinsic matrix.

    Rotation order: Yaw (Y-axis) → Pitch (X-axis) → Roll (Z-axis)
    Camera mounting conventions:
      - Mounted at `height` metres above the floor
      - Pointing `yaw_deg` degrees from forward (-ve = left, +ve = right)
      - Tilted `pitch_deg` degrees vertically (-ve = down)
      - Rolled `roll_deg` degrees (-ve = left lean)
    """
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)
    y = np.radians(yaw_deg)

    # Rotation about X (pitch)
    Rp = np.array(
        [
            [1, 0, 0],
            [0, np.cos(p), -np.sin(p)],
            [0, np.sin(p), np.cos(p)],
        ],
        dtype=np.float32,
    )

    # Rotation about Z (roll)
    Rr = np.array(
        [
            [np.cos(r), -np.sin(r), 0],
            [np.sin(r), np.cos(r), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Rotation about Y (yaw — pointing direction)
    Ry = np.array(
        [
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)],
        ],
        dtype=np.float32,
    )

    R = Ry @ Rp @ Rr  # yaw first, then pitch, then roll
    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = R
    E[:3, 3] = [0, -height, 0]
    return E
