"""
Panoramic stitching of 3 camera feeds using rotation homography.

Because we know the exact yaw angle between cameras, no feature matching is
needed. The homography for each camera is:

    H = K_output @ Ry(yaw_deg) @ K_cam⁻¹

This warps each camera's image onto a shared wide canvas as if they were all
taken from the same position but by a single wide-angle camera.

Limitation: objects closer than ~1 m show parallax artifacts (cameras are
physically separated). Objects >1 m away stitch cleanly.
"""

import cv2
import numpy as np


def render_panorama(
    cam_frames, cap_w: int = 424, cap_h: int = 240, out_w: int = 672, out_h: int = 384
):
    """
    Stitch 3 camera frames into a panoramic view.

    Parameters
    ----------
    cam_frames : list of (img_np, intr, yaw_deg)
        img_np   — (out_h, out_w, 3) uint8 RGB image (already resized to model res)
        intr     — pyrealsense2 intrinsics at capture resolution (cap_w × cap_h)
        yaw_deg  — float, camera yaw in degrees

    Returns
    -------
    (out_h, canvas_w, 3) uint8 RGB panoramic image
    """
    if not cam_frames:
        return None

    sx = out_w / cap_w  # ≈ 1.585 for 672/424
    sy = out_h / cap_h  # ≈ 1.6   for 384/240

    cameras = []
    center_fx = out_w * 0.85  # fallback if no intr
    for img, intr, yaw in cam_frames:
        if intr is not None:
            fx = intr.fx * sx
            fy = intr.fy * sy
            cx = intr.ppx * sx
            cy = intr.ppy * sy
        else:
            fx = fy = out_w * 0.85
            cx, cy = out_w / 2.0, out_h / 2.0
        if abs(yaw) < 5:
            center_fx = fx
        cameras.append((img, fx, fy, cx, cy, float(yaw)))

    # Canvas is 2.5× wider so all three cameras fit.
    # The center camera's principal point maps to canvas centre.
    canvas_w = int(out_w * 2.5)
    canvas_h = out_h
    cx_out = canvas_w / 2.0
    cy_out = out_h / 2.0

    K_out = np.float64(
        [
            [center_fx, 0, cx_out],
            [0, center_fx, cy_out],
            [0, 0, 1],
        ]
    )

    # Each camera gets a Cauchy weight centred on its optical-axis projection.
    # Side cameras drawn first; centre camera drawn last (highest priority).
    acc_blend = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    acc_weights = np.zeros((canvas_h, canvas_w), dtype=np.float64)

    x_coords = np.arange(canvas_w, dtype=np.float64)
    sigma = out_w / 2.0  # blend width ~half a camera FOV

    sorted_cams = sorted(cameras, key=lambda c: abs(c[5]), reverse=True)

    for img, fx, fy, cx, cy, yaw in sorted_cams:
        K_src = np.float64(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )

        y = np.radians(yaw)
        Ry = np.float64(
            [
                [np.cos(y), 0, np.sin(y)],
                [0, 1, 0],
                [-np.sin(y), 0, np.cos(y)],
            ]
        )

        # Map source pixels → canvas pixels
        H = K_out @ Ry @ np.linalg.inv(K_src)
        warped = cv2.warpPerspective(
            img,
            H,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Optical-axis position on canvas (for blending weight)
        oa = K_out @ (Ry @ np.array([0.0, 0.0, 1.0]))
        x_oa = oa[0] / oa[2]

        w_1d = 1.0 / (1.0 + ((x_coords - x_oa) / sigma) ** 2)
        w_2d = np.tile(w_1d[np.newaxis, :], (canvas_h, 1))
        w_2d *= (np.sum(warped, axis=2) > 0).astype(np.float64)

        acc_blend += warped.astype(np.float64) * w_2d[:, :, np.newaxis]
        acc_weights += w_2d

    # Normalise and convert
    valid = acc_weights > 1e-6
    result = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    result[valid] = np.clip(
        acc_blend[valid] / acc_weights[valid, np.newaxis], 0, 255
    ).astype(np.uint8)

    # Draw faint vertical lines where each camera's FOV transitions.
    for _, fx, _, cx_s, _, yaw in cameras:
        # Where does the EDGE of this camera's FOV fall on the canvas?
        for edge_u in [0.0, float(out_w)]:
            d = np.array([(edge_u - cx_s) / fx, 0.0, 1.0])
            y = np.radians(yaw)
            Ry_e = np.float64(
                [[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]
            )
            d_w = Ry_e @ d
            if d_w[2] < 0.01:
                continue
            x_e = int(K_out[0, 0] * d_w[0] / d_w[2] + K_out[0, 2])
            if 0 < x_e < canvas_w:
                result[:, max(0, x_e - 1) : x_e + 1] = (
                    (result[:, max(0, x_e - 1) : x_e + 1].astype(np.int16) + 30)
                    .clip(0, 255)
                    .astype(np.uint8)
                )

    for _, fx, _, cx_s, _, yaw in cameras:
        # Label at optical axis position
        d_oa = np.array([0.0, 0.0, 1.0])
        y_r = np.radians(yaw)
        Ry_l = np.float64(
            [[np.cos(y_r), 0, np.sin(y_r)], [0, 1, 0], [-np.sin(y_r), 0, np.cos(y_r)]]
        )
        d_w = Ry_l @ d_oa
        if d_w[2] < 0.01:
            continue
        x_l = int(K_out[0, 0] * d_w[0] / d_w[2] + K_out[0, 2])
        if 5 < x_l < canvas_w - 80:
            label = "L" if yaw < -5 else ("R" if yaw > 5 else "C")
            label += f" {int(yaw):+d}deg"
            pos = (x_l - 20, 20)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.45
            # 1. Thick dark outline → readable on any background
            cv2.putText(result, label, pos, font, scale, (0, 0, 0), 4, cv2.LINE_AA)
            # 2. Bright cyan text on top
            cv2.putText(result, label, pos, font, scale, (0, 230, 200), 1, cv2.LINE_AA)

    return result
