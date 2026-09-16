"""Free-space classification and BEV visualization."""

import numpy as np
from PIL import Image, ImageDraw

from .. import config


def compute_fov_mask(bev_result, yaw_deg=0.0):
    """
    Compute which BEV cells fall within the camera's horizontal FOV.

    yaw_deg=0   → forward cone centred at BEV centre (single camera)
    yaw_deg=±45 → cone shifted left/right (multi-camera)
    """
    bev_h, bev_w = bev_result.bev_h, bev_result.bev_w
    x_min, x_max = bev_result.x_range
    z_min, z_max = bev_result.z_range
    half_fov = np.arctan2(config.IMG_W / 2.0, bev_result.fx)
    yaw_rad = np.radians(yaw_deg)

    fov_mask = np.zeros((bev_h, bev_w), dtype=bool)
    for row in range(bev_h):
        z_world = z_min + (1.0 - row / bev_h) * (z_max - z_min)
        if z_world <= 0.1:
            continue
        # Centre column of this camera's FOV cone at this depth
        x_center = z_world * np.tan(yaw_rad)
        col_center = (x_center - x_min) / (x_max - x_min) * bev_w
        # Half-width at this depth (slightly wider for yawed cameras)
        cos_y = max(abs(np.cos(yaw_rad)), 0.15)
        spread = z_world * np.tan(half_fov) / cos_y
        half_px = spread / (x_max - x_min) * bev_w
        lo = max(0, int(col_center - half_px))
        hi = min(bev_w, int(col_center + half_px))
        if lo < hi:
            fov_mask[row, lo:hi] = True
    return fov_mask


def classify_freespace(bev_result, fov_mask):
    """
    Classify BEV cells as free, obstacle, or unknown.

    Returns:
        obstacle_mask: (H, W) bool — True for obstacle cells
        free_mask:     (H, W) bool — True for free cells
        unknown_mask:  (H, W) bool — True for unknown cells
    """
    occupied = bev_result.occupied
    min_h = bev_result.min_height
    h_range = bev_result.height_range

    if occupied.any():
        ground_level = np.percentile(bev_result.max_height[bev_result.occupied], 75)
    else:
        ground_level = 0.0

    is_elevated = min_h < (ground_level - config.OBSTACLE_HEIGHT_THRESH)
    is_tall = h_range > config.OBSTACLE_RANGE_THRESH

    obstacle = fov_mask & occupied & (is_elevated | is_tall)
    free = fov_mask & occupied & ~obstacle
    unknown = fov_mask & ~occupied

    return obstacle, free, unknown


def render_freespace(
    bev_result,
    obstacle,
    free,
    unknown,
    fov_mask,
    inflated=None,
    path_cells=None,
    show_grid=True,
):
    """Render a colorised free-space BEV image."""
    C = config.COLORS
    bev_h, bev_w = bev_result.bev_h, bev_result.bev_w
    z_min, z_max = bev_result.z_range
    x_min, x_max = bev_result.x_range

    fs = np.full((bev_h, bev_w, 3), C["outside_fov"], dtype=np.uint8)
    fs[fov_mask] = C["fov_bg"]
    fs[unknown] = C["unknown"]
    fs[free] = C["free"]

    if inflated is not None:
        inflate_only = inflated & ~obstacle & fov_mask
        fs[inflate_only] = C["inflated"]

    fs[obstacle] = C["obstacle"]

    obs_colors = bev_result.color_image.copy()
    blend = 0.5
    fs[obstacle] = (
        blend * fs[obstacle].astype(np.float32)
        + (1 - blend) * obs_colors[obstacle].astype(np.float32)
    ).astype(np.uint8)

    if path_cells:
        for r, c in path_cells:
            if 0 <= r < bev_h and 0 <= c < bev_w:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < bev_h and 0 <= cc < bev_w:
                            fs[rr, cc] = C["path"]

    pil = Image.fromarray(fs)
    draw = ImageDraw.Draw(pil)

    if show_grid:
        for dist in range(10, int(z_max), 10):
            row_px = int((1.0 - (dist - z_min) / (z_max - z_min)) * bev_h)
            if 0 <= row_px < bev_h:
                draw.line([(0, row_px), (bev_w, row_px)], fill=(50, 50, 55), width=1)
                draw.text((4, row_px + 2), f"{dist}m", fill=(90, 90, 90))
        for lat in range(-10, 11, 10):
            col_px = int((lat - x_min) / (x_max - x_min) * bev_w)
            if 0 < col_px < bev_w:
                draw.line([(col_px, 0), (col_px, bev_h)], fill=(40, 40, 45), width=1)

    # Ego marker
    cx = bev_w // 2
    cy = bev_h - 22
    draw.polygon([(cx, cy - 10), (cx - 6, cy + 4), (cx + 6, cy + 4)], fill=C["path"])
    draw.text((cx + 10, cy - 6), "EGO", fill=C["path"])

    # Legend
    y0 = 10
    items = [
        ("Free", C["free"]),
        ("Occupied", C["obstacle"]),
        ("Unknown", C["unknown"]),
        ("Outside FOV", C["outside_fov"]),
    ]
    if inflated is not None:
        items.insert(2, ("Inflated", C["inflated"]))
    if path_cells:
        items.append(("Path", C["path"]))
    for label, color in items:
        draw.rectangle(
            [(bev_w - 115, y0), (bev_w - 100, y0 + 10)],
            fill=color,
            outline=(120, 120, 120),
        )
        draw.text((bev_w - 95, y0 - 1), label, fill=(160, 160, 160))
        y0 += 16

    return np.array(pil)


def render_color_bev(bev_result, path_world=None, show_grid=True):
    """Render the color BEV with optional grid overlays and path."""
    bev_rgb = bev_result.color_image.copy()
    bev_h, bev_w = bev_result.bev_h, bev_result.bev_w
    z_min, z_max = bev_result.z_range
    x_min, x_max = bev_result.x_range

    pil = Image.fromarray(bev_rgb)
    draw = ImageDraw.Draw(pil)

    if show_grid:
        for dist in range(10, int(z_max), 10):
            row_px = int((1.0 - (dist - z_min) / (z_max - z_min)) * bev_h)
            if 0 <= row_px < bev_h:
                draw.line([(0, row_px), (bev_w, row_px)], fill=(60, 60, 60), width=1)
                draw.text((4, row_px + 2), f"{dist}m", fill=(80, 80, 80))
        for lat in range(-10, 11, 10):
            col_px = int((lat - x_min) / (x_max - x_min) * bev_w)
            if 0 < col_px < bev_w:
                draw.line([(col_px, 0), (col_px, bev_h)], fill=(40, 40, 40), width=1)

    cx = bev_w // 2
    cy = bev_h - 22
    draw.polygon(
        [(cx, cy - 10), (cx - 6, cy + 4), (cx + 6, cy + 4)], fill=config.COLORS["path"]
    )
    draw.text((cx + 10, cy - 6), "EGO", fill=config.COLORS["path"])

    if path_world:
        pts = []
        for xw, zw in path_world:
            col = int((xw - x_min) / (x_max - x_min) * bev_w)
            row = int((1.0 - (zw - z_min) / (z_max - z_min)) * bev_h)
            pts.append((col, row))
        if len(pts) > 1:
            draw.line(pts, fill=config.COLORS["path"], width=3)
        for p in [pts[0], pts[-1]]:
            draw.ellipse(
                [p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=config.COLORS["path"]
            )

    return np.array(pil)
