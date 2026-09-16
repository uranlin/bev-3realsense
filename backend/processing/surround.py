"""Ground-plane inverse perspective rendering. Elevated objects are distorted."""

import cv2
import numpy as np

_BG = (15, 18, 25)  # outside all FOVs
_GRID_LINE = (38, 44, 58)
_GRID_LABEL = (75, 85, 100)
_SEAM_LINE = (50, 60, 80)
_EGO_FILL = (34, 197, 94)  # green
_EGO_BORDER = (52, 211, 153)
_EGO_TEXT = (10, 20, 15)
_ARROW = (52, 211, 153)
_FOV_ARC = (45, 55, 75)


def _w2b(X: float, Z: float, x_min, x_max, z_min, z_max, bev_h, bev_w):
    """World (X, Z) → BEV pixel (col, row)."""
    col = int((x_max - X) / (x_max - x_min) * bev_w)
    row = int((z_max - Z) / (z_max - z_min) * bev_h)
    return col, row


def _draw_grid(bev, x_min, x_max, z_min, z_max, bev_h, bev_w):
    """Concentric distance lines + lateral centre line."""
    step = 1.0
    d = step
    while d <= z_max + 0.1:
        _, row = _w2b(0, d, x_min, x_max, z_min, z_max, bev_h, bev_w)
        if 0 < row < bev_h:
            cv2.line(bev, (0, row), (bev_w, row), _GRID_LINE, 1, cv2.LINE_AA)
            cv2.putText(
                bev,
                f"{d:.0f}m",
                (6, row - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                _GRID_LABEL,
                1,
                cv2.LINE_AA,
            )
        d += step
    # Centre line
    cx = bev_w // 2
    cv2.line(bev, (cx, 0), (cx, bev_h), _GRID_LINE, 1, cv2.LINE_AA)


def _draw_ego(bev, x_min, x_max, z_min, z_max, bev_h, bev_w, body_w=0.35, body_l=0.55):
    """Car-shaped EGO marker with direction arrow."""
    # Body rectangle (centred at world 0,0 but sitting above EGO — z≈0)
    hw, hl = body_w / 2, body_l / 2
    corners = [(-hw, hl), (hw, hl), (hw, -hl), (-hw, -hl)]
    pts_bev = [_w2b(x, z, x_min, x_max, z_min, z_max, bev_h, bev_w) for x, z in corners]
    pts = np.array(pts_bev, dtype=np.int32)

    cv2.fillPoly(bev, [pts], _EGO_FILL)
    cv2.polylines(bev, [pts], True, _EGO_BORDER, 2, cv2.LINE_AA)

    # "Windscreen" line (front of vehicle)
    wl = _w2b(-hw * 0.7, hl * 0.5, x_min, x_max, z_min, z_max, bev_h, bev_w)
    wr = _w2b(hw * 0.7, hl * 0.5, x_min, x_max, z_min, z_max, bev_h, bev_w)
    cv2.line(bev, wl, wr, _EGO_BORDER, 1, cv2.LINE_AA)

    # Direction arrow
    cx, cy = _w2b(0, 0, x_min, x_max, z_min, z_max, bev_h, bev_w)
    _, tip = _w2b(0, hl + 0.2, x_min, x_max, z_min, z_max, bev_h, bev_w)
    cv2.arrowedLine(bev, (cx, cy), (cx, tip), _ARROW, 2, cv2.LINE_AA, tipLength=0.5)

    # Label
    cv2.putText(
        bev,
        "EGO",
        (cx - 13, cy + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        _EGO_TEXT,
        1,
        cv2.LINE_AA,
    )


def _draw_fov_arcs(bev, cam_data, x_min, x_max, z_min, z_max, bev_h, bev_w):
    """
    Draw subtle arcs at each camera's maximum range showing its FOV cone.
    Also draw thin radial lines at the FOV edges.
    """
    for _, K, _, yaw_deg in cam_data:
        fx = float(K[0, 0])
        img_w = float(K[0, 2]) * 2  # cx ≈ W/2
        half_fov = np.arctan2(img_w / 2.0, fx)
        yaw_rad = np.radians(yaw_deg)

        angles = np.linspace(yaw_rad - half_fov, yaw_rad + half_fov, 40)
        pts = []
        for a in angles:
            X = float(z_max) * np.sin(a)
            Z = float(z_max) * np.cos(a)
            pts.append(_w2b(X, Z, x_min, x_max, z_min, z_max, bev_h, bev_w))

        for i in range(len(pts) - 1):
            cv2.line(bev, pts[i], pts[i + 1], _FOV_ARC, 1, cv2.LINE_AA)

        # Radial boundary lines
        cx, cz = _w2b(0, 0, x_min, x_max, z_min, z_max, bev_h, bev_w)
        for angle in [yaw_rad - half_fov, yaw_rad + half_fov]:
            X = float(z_max) * np.sin(angle)
            Z = float(z_max) * np.cos(angle)
            pe = _w2b(X, Z, x_min, x_max, z_min, z_max, bev_h, bev_w)
            cv2.line(bev, (cx, cz), pe, _FOV_ARC, 1, cv2.LINE_AA)


def _add_vignette(bev):
    """Apply a subtle radial vignette — edges slightly darker."""
    bev_h, bev_w = bev.shape[:2]
    cx, cy = bev_w / 2.0, bev_h / 2.0
    Y, X = np.ogrid[:bev_h, :bev_w]
    r = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    mask = np.clip(1.0 - 0.25 * r, 0.6, 1.0).astype(np.float32)
    bev[:] = np.clip(bev.astype(np.float32) * mask[:, :, np.newaxis], 0, 255).astype(
        np.uint8
    )


def render_surround_bev(
    cam_data,
    bev_h: int = 500,
    bev_w: int = 500,
    x_min: float = -4.0,
    x_max: float = 4.0,
    z_min: float = 0.3,
    z_max: float = 5.0,
    show_grid: bool = True,
):
    """
    Generate an automotive-style surround BEV via IPM.

    Parameters
    ----------
    cam_data : list of (img_np, K, E, yaw_deg)
    bev_h, bev_w : output BEV dimensions (pixels)
    x_min, x_max : lateral world range (metres)
    z_min, z_max : depth world range (metres)

    Returns
    -------
    (bev_h, bev_w, 3) uint8 RGB
    """
    if not cam_data:
        return None

    N = bev_h * bev_w

    # World ground coordinates for every BEV pixel
    x_lin = np.linspace(x_max, x_min, bev_w, dtype=np.float32)
    z_lin = np.linspace(z_max, z_min, bev_h, dtype=np.float32)
    X_g, Z_g = np.meshgrid(x_lin, z_lin)
    P_world = np.stack(
        [X_g.ravel(), np.zeros(N, np.float32), Z_g.ravel()], axis=0
    )  # (3, N)

    acc_rgb = np.zeros((N, 3), dtype=np.float64)
    acc_w = np.zeros(N, dtype=np.float64)

    for img_np, K, E, yaw_deg in cam_data:
        img_h, img_w = img_np.shape[:2]
        R_d = E[:3, :3].astype(np.float64)
        t_d = E[:3, 3].astype(np.float64)
        K_d = K.astype(np.float64)

        # Back-project to camera space
        P_cam = R_d.T @ (P_world.astype(np.float64) - t_d[:, None])

        z_c = P_cam[2]
        front = z_c > 0.05

        # Project to image pixels
        u = K_d[0, 0] * P_cam[0] / np.where(front, z_c, 1.0) + K_d[0, 2]
        v = K_d[1, 1] * P_cam[1] / np.where(front, z_c, 1.0) + K_d[1, 2]

        valid = front & (u >= 0) & (u < img_w - 1) & (v >= 0) & (v < img_h - 1)

        # Confidence weight:
        # (a) angular distance from camera's optical axis (smaller = better)
        cx_, cy_ = K_d[0, 2], K_d[1, 2]
        fx_, fy_ = K_d[0, 0], K_d[1, 1]
        ang_x = np.arctan(np.abs(np.where(valid, u - cx_, 0.0)) / fx_)
        ang_y = np.arctan(np.abs(np.where(valid, v - cy_, 0.0)) / fy_)
        ang = np.sqrt(ang_x**2 + ang_y**2)
        w_ang = np.exp(-3.0 * (ang / np.radians(35)) ** 2)  # half-FOV ≈ 35°

        # (b) depth (closer = sharper texture)
        w_dep = 1.0 / (1.0 + (z_c / z_max) ** 1.5)

        weight = w_ang * w_dep * valid.astype(np.float64)

        # Bilinear sample via cv2.remap
        u_map = u.reshape(bev_h, bev_w).astype(np.float32)
        v_map = v.reshape(bev_h, bev_w).astype(np.float32)
        u_map[~valid.reshape(bev_h, bev_w)] = -1.0
        v_map[~valid.reshape(bev_h, bev_w)] = -1.0

        sampled = cv2.remap(
            img_np,
            u_map,
            v_map,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        acc_rgb += sampled.reshape(N, 3).astype(np.float64) * weight[:, None]
        acc_w += weight

    # Normalise
    covered = acc_w > 1e-3
    out = np.full((N, 3), _BG, dtype=np.uint8)
    out[covered] = np.clip(acc_rgb[covered] / acc_w[covered, None], 0, 255).astype(
        np.uint8
    )
    bev = out.reshape(bev_h, bev_w, 3)

    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], np.float32)
    sharp = cv2.filter2D(bev, -1, kernel)
    bev = np.where(covered.reshape(bev_h, bev_w, 1), sharp, bev)

    _add_vignette(bev)
    if show_grid:
        _draw_grid(bev, x_min, x_max, z_min, z_max, bev_h, bev_w)
        _draw_fov_arcs(bev, cam_data, x_min, x_max, z_min, z_max, bev_h, bev_w)
    _draw_ego(bev, x_min, x_max, z_min, z_max, bev_h, bev_w)

    return bev
