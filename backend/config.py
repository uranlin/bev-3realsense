"""Centralized configuration for the BEV navigation demo."""

# Keep the geometry-only runtime independent of PyTorch/OpenMP.
import os

DEVICE = os.environ.get("BEV_DEVICE", "cpu")

# Image processing
IMG_H, IMG_W = 384, 672  # Working resolution; model adapters handle preprocessing.

# BEV grid — INDOOR preset
BEV_SIZE = (500, 500)
BEV_X_RANGE = (-4.0, 4.0)  # ±4m lateral for room-scale
BEV_Z_RANGE = (0.3, 5.0)  # matches DEPTH_MAX

# Depth — INDOOR
DEPTH_MIN, DEPTH_MAX = 0.3, 5.0
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

# Intel RealSense D435i intrinsics (approximate, at 640×480 native resolution)
# These get scaled automatically in make_intrinsic() based on IMG_W × IMG_H.
# Check your exact values in RealSense Viewer: Depth > Advanced > Intrinsics.
# Override at runtime via the Settings panel (fx/fy/cx/cy fields).
REALSENSE_FX = 617.4  # horizontal focal length (pixels at 640×480)
REALSENSE_FY = 617.4  # vertical focal length
REALSENSE_CX = 321.6  # principal point X
REALSENSE_CY = 241.1  # principal point Y

# Free-space classification — tighter thresholds for indoor (shorter range)
OBSTACLE_HEIGHT_THRESH = 0.2  # metres above ground → obstacle (lower for indoor)
OBSTACLE_RANGE_THRESH = 0.4  # vertical extent → obstacle

# Path planning
ROBOT_RADIUS_M = 0.25  # smaller inflation for narrow corridors

# Freespace colors (RGB)
COLORS = {
    "outside_fov": (12, 12, 12),
    "fov_bg": (22, 22, 28),
    "unknown": (40, 50, 85),
    "free": (20, 110, 55),
    "obstacle": (200, 50, 50),
    "inflated": (120, 80, 30),
    "path": (0, 212, 170),
}
