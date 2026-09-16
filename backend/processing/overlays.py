"""Object detection overlays for BEV images."""

import numpy as np

DETECTION_COLORS = {
    "person": (255, 100, 40),
    "car": (40, 220, 80),
    "truck": (40, 200, 120),
    "bus": (40, 180, 180),
    "bicycle": (100, 180, 255),
    "motorcycle": (180, 100, 255),
    "tv": (255, 200, 40),
    "laptop": (255, 220, 80),
    "keyboard": (80, 255, 200),
    "mouse": (80, 220, 255),
    "cup": (255, 140, 200),
    "chair": (200, 140, 255),
    "default": (180, 180, 180),
}


def draw_bev_detections(bev_rgb, bev_dets, x_range, z_range, bev_size):
    """
    Project camera detections into BEV space and draw markers.

    Parameters
    ----------
    bev_rgb   : (H, W, 3) uint8 RGB numpy array  — modified in-place
    bev_dets  : list of (det_dict, K_33, E_44)
    x_range   : (x_min, x_max)
    z_range   : (z_min, z_max)
    bev_size  : (bev_h, bev_w)
    """
    import cv2 as _cv2

    bev_h, bev_w = bev_size
    x_min, x_max = x_range
    z_min, z_max = z_range
    canvas = _cv2.cvtColor(bev_rgb, _cv2.COLOR_RGB2BGR)
    for det, K, E in bev_dets:
        dist = det.get("distance")
        if not dist or dist <= 0:
            continue
        bbox = det["bbox"]
        u_c = (bbox[0] + bbox[2]) / 2.0
        v_c = (bbox[1] + bbox[3]) / 2.0
        fx, fy = (float(K[0, 0]), float(K[1, 1]))
        cx, cy = (float(K[0, 2]), float(K[1, 2]))
        X_cam = (u_c - cx) * dist / fx
        Y_cam = (v_c - cy) * dist / fy
        Z_cam = float(dist)
        P_world = E @ np.array([X_cam, Y_cam, Z_cam, 1.0], dtype=np.float64)
        X_w, Z_w = (float(P_world[0]), float(P_world[2]))
        col = int((X_w - x_min) / (x_max - x_min) * bev_w)
        row = int((z_max - Z_w) / (z_max - z_min) * bev_h)
        col = max(14, min(bev_w - 15, col))
        row = max(14, min(bev_h - 15, row))
        rgb = DETECTION_COLORS.get(det["class"], DETECTION_COLORS["default"])
        bgr = (rgb[2], rgb[1], rgb[0])
        _cv2.circle(canvas, (col, row), 14, bgr, 1, _cv2.LINE_AA)
        _cv2.circle(canvas, (col, row), 8, (0, 0, 0), -1)
        _cv2.circle(canvas, (col, row), 7, bgr, -1, _cv2.LINE_AA)
        label = f"{det['class']} {dist:.1f}m"
        lx, ly = (col + 11, row + 4)
        _cv2.putText(
            canvas,
            label,
            (lx, ly),
            _cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 0, 0),
            3,
            _cv2.LINE_AA,
        )
        _cv2.putText(
            canvas,
            label,
            (lx, ly),
            _cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            bgr,
            1,
            _cv2.LINE_AA,
        )
    bev_rgb[:] = _cv2.cvtColor(canvas, _cv2.COLOR_BGR2RGB)
