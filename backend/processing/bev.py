"""Project RGB-D observations into a metric grid and fuse cell measurements."""

import numpy as np

from .. import config

# GPU scatter is explicitly enabled; CPU geometry never imports torch.
_TORCH_OK = False
if str(config.DEVICE).startswith("cuda"):
    try:
        import torch as _torch

        _GPU = _torch.device(config.DEVICE)
        _TORCH_OK = _torch.cuda.is_available()
    except ImportError:
        pass


def _scatter_gpu(colors_np, valid, v_ix, v_iz, y_w, bev_h, bev_w):
    """GPU scatter accumulation using torch.scatter_add_."""
    import torch

    device = _GPU
    n = bev_h * bev_w

    idx = torch.from_numpy((v_iz * bev_w + v_ix).astype(np.int64)).to(device)
    col = torch.from_numpy(colors_np[valid].astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_w[valid].astype(np.float32)).to(device)

    bev_acc = torch.zeros(n, 3, device=device)
    bev_cnt = torch.zeros(n, device=device)
    bev_acc.scatter_add_(0, idx.unsqueeze(1).expand(-1, 3), col)
    bev_cnt.scatter_add_(0, idx, torch.ones(len(idx), device=device))

    # Height extremes
    min_h = torch.full((n,), 999.0, device=device)
    max_h = torch.full((n,), -999.0, device=device)
    try:  # requires PyTorch ≥ 1.12
        min_h = min_h.scatter_reduce(0, idx, y_t, reduce="amin", include_self=True)
        max_h = max_h.scatter_reduce(0, idx, y_t, reduce="amax", include_self=True)
    except Exception:
        # Fallback: approximate with scatter_add on absolute value
        min_h_np = np.full(n, 999.0, dtype=np.float32)
        max_h_np = np.full(n, -999.0, dtype=np.float32)
        np.minimum.at(min_h_np, v_iz * bev_w + v_ix, y_w[valid].astype(np.float32))
        np.maximum.at(max_h_np, v_iz * bev_w + v_ix, y_w[valid].astype(np.float32))
        min_h = torch.from_numpy(min_h_np).to(device)
        max_h = torch.from_numpy(max_h_np).to(device)

    occupied = bev_cnt > 0
    safe_cnt = bev_cnt.clamp(min=1).unsqueeze(1)
    bev_rgb = torch.full((n, 3), 20, dtype=torch.uint8, device=device)
    bev_rgb[occupied] = (bev_acc[occupied] / safe_cnt[occupied]).clamp(0, 255).byte()

    bev_rgb_np = bev_rgb.reshape(bev_h, bev_w, 3).cpu().numpy()
    occupied_np = occupied.reshape(bev_h, bev_w).cpu().numpy()
    min_h_np = min_h.reshape(bev_h, bev_w).cpu().numpy()
    max_h_np = max_h.reshape(bev_h, bev_w).cpu().numpy()
    cnt_np = bev_cnt.reshape(bev_h, bev_w).cpu().numpy()

    return bev_rgb_np, occupied_np, min_h_np, max_h_np, cnt_np


def _scatter_cpu(colors_np, valid, v_ix, v_iz, y_w, bev_h, bev_w):
    """CPU fallback scatter accumulation."""
    bev_acc = np.zeros((bev_h, bev_w, 3), dtype=np.float64)
    bev_cnt = np.zeros((bev_h, bev_w), dtype=np.float64)
    min_h = np.full((bev_h, bev_w), 999.0, dtype=np.float64)
    max_h = np.full((bev_h, bev_w), -999.0, dtype=np.float64)

    np.add.at(bev_acc, (v_iz, v_ix), colors_np[valid])
    np.add.at(bev_cnt, (v_iz, v_ix), 1.0)
    np.minimum.at(min_h, (v_iz, v_ix), y_w[valid])
    np.maximum.at(max_h, (v_iz, v_ix), y_w[valid])

    occupied = bev_cnt > 0
    bev_rgb = np.full((bev_h, bev_w, 3), 20, dtype=np.uint8)
    bev_rgb[occupied] = (bev_acc[occupied] / bev_cnt[occupied, np.newaxis]).astype(
        np.uint8
    )

    return bev_rgb, occupied, min_h, max_h, bev_cnt


class BEVProjection:
    """Project image + depth into a Bird's Eye View grid."""

    def __init__(self):
        self.bev_h, self.bev_w = config.BEV_SIZE
        self.x_range = config.BEV_X_RANGE
        self.z_range = config.BEV_Z_RANGE

    def project(self, img_np, depth_np, K, E):
        self.bev_h, self.bev_w = config.BEV_SIZE
        self.x_range = config.BEV_X_RANGE
        self.z_range = config.BEV_Z_RANGE

        H, W = depth_np.shape
        bev_h, bev_w = self.bev_h, self.bev_w
        x_min, x_max = self.x_range
        z_min, z_max = self.z_range

        uu, vv = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
        )
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        valid_depth = (
            np.isfinite(depth_np)
            & (depth_np >= config.DEPTH_MIN)
            & (depth_np <= config.DEPTH_MAX)
        )
        z_c = np.where(valid_depth, depth_np, 0.0)
        x_c = (uu - cx) * z_c / fx
        y_c = (vv - cy) * z_c / fy

        N = H * W
        pts_cam = np.stack(
            [x_c.ravel(), y_c.ravel(), z_c.ravel(), np.ones(N, dtype=np.float32)],
            axis=1,
        )
        pts_world = (E @ pts_cam.T).T[:, :3]

        x_w, y_w, z_w = pts_world[:, 0], pts_world[:, 1], pts_world[:, 2]

        ix = np.floor((x_w - x_min) / (x_max - x_min) * bev_w).astype(np.int32)
        iz = np.floor((z_w - z_min) / (z_max - z_min) * bev_h).astype(np.int32)

        valid = (ix >= 0) & (ix < bev_w) & (iz >= 0) & (iz < bev_h)
        valid &= (y_w > -4.0) & (y_w < 4.0)
        valid &= valid_depth.ravel()

        v_ix = ix[valid]
        v_iz = iz[valid]

        colors = img_np.reshape(N, 3).astype(np.float32)

        if _TORCH_OK and _GPU.type == "cuda" and len(v_ix) > 100:
            bev_rgb, occupied, min_h, max_h, cnt = _scatter_gpu(
                colors, valid, v_ix, v_iz, y_w, bev_h, bev_w
            )
        else:
            bev_rgb, occupied, min_h, max_h, cnt = _scatter_cpu(
                colors, valid, v_ix, v_iz, y_w, bev_h, bev_w
            )

        # Flip so forward = top of BEV image
        bev_rgb = np.flipud(bev_rgb)
        occupied = np.flipud(occupied)
        min_h = np.flipud(min_h)
        max_h = np.flipud(max_h)
        cnt = np.flipud(cnt)

        height_range = np.zeros((bev_h, bev_w), dtype=np.float64)
        height_range[occupied] = max_h[occupied] - min_h[occupied]

        return BEVResult(
            color_image=bev_rgb,
            occupied=occupied,
            min_height=min_h,
            max_height=max_h,
            height_range=height_range,
            count=cnt,
            fx=fx,
            bev_h=bev_h,
            bev_w=bev_w,
            x_range=self.x_range,
            z_range=self.z_range,
        )


class BEVResult:
    """Container for BEV projection outputs."""

    __slots__ = [
        "color_image",
        "occupied",
        "min_height",
        "max_height",
        "height_range",
        "count",
        "fx",
        "bev_h",
        "bev_w",
        "x_range",
        "z_range",
    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def fuse_results(results):
    """Fuse observations, preserving black pixels and height extrema."""
    if not results:
        raise ValueError("At least one BEV result is required")
    first = results[0]
    count = np.zeros_like(first.count, dtype=np.float64)
    acc = np.zeros_like(first.color_image, dtype=np.float64)
    low = np.full(first.count.shape, 999.0)
    high = np.full(first.count.shape, -999.0)
    for item in results:
        if (
            item.count.shape != count.shape
            or item.x_range != first.x_range
            or item.z_range != first.z_range
        ):
            raise ValueError("BEV grids must share shape and metric bounds")
        count += item.count
        acc += item.color_image * item.count[..., None]
        low = np.minimum(low, np.where(item.occupied, item.min_height, 999.0))
        high = np.maximum(high, np.where(item.occupied, item.max_height, -999.0))
    observed = count > 0
    color = np.full_like(first.color_image, 20)
    color[observed] = (acc[observed] / count[observed, None]).astype(np.uint8)
    return BEVResult(
        color_image=color,
        occupied=observed,
        min_height=low,
        max_height=high,
        height_range=np.where(observed, high - low, 0),
        count=count,
        fx=first.fx,
        bev_h=first.bev_h,
        bev_w=first.bev_w,
        x_range=first.x_range,
        z_range=first.z_range,
    )
