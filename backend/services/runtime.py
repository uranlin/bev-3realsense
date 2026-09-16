"""Owns camera resources, inference adapters and the current navigation frame."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from PIL import Image
from scipy.ndimage import zoom as ndim_zoom

from backend import config
from backend.api.schemas import CameraConfig, MultiCamConfig
from backend.models.detector import ObjectDetector
from backend.models.realsense import RealSenseCapture
from backend.models.realsense_multi import MultiRealSenseCapture, list_connected_serials
from backend.processing import (
    BEVProjection,
    classify_freespace,
    compute_fov_mask,
    inflate_obstacles,
    render_color_bev,
    render_freespace,
)
from backend.processing.bev import fuse_results
from backend.processing.overlays import draw_bev_detections
from backend.utils import encode_image, make_extrinsic, make_intrinsic, turbo_colormap
from backend.utils.camera import scale_intrinsic

DEPTH_MODELS = {
    "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
}
logger = logging.getLogger("bev-nav")


class NavigationRuntime:
    def __init__(self):
        self.current_depth_model_key = "indoor"
        self.depth_estimator = None
        self.realsense = None
        self.multi_rs = None
        self.detector = None
        self.bev_projector = BEVProjection()
        self.intrinsic_np = None
        self.extrinsic_np = None
        self.latest_bev = None
        self.latest_obstacle = None
        self.latest_inflated = None
        self.latest_fov = None
        self.latest_free = None
        self.mode = os.environ.get("BEV_MODE", "demo")
        self._yolo_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="yolo"
        )
        self._yolo_cache = {}
        self._yolo_lock = threading.Lock()
        self.current_cam = CameraConfig()
        self.multi_cam_cfg = MultiCamConfig()

    def _run_yolo_bg(self, label: str, img_np, depth_r):
        """Background YOLO inference — updates cache when done."""
        if self.detector is None:
            return
        try:
            dets, annotated = self.detector.detect(img_np, depth_r)
            with self._yolo_lock:
                self._yolo_cache[label] = (dets, annotated)
        except Exception as e:
            logger.debug(f"Async YOLO {label}: {e}")

    def rebuild_camera(self):
        self.intrinsic_np = make_intrinsic(
            self.current_cam.fx,
            self.current_cam.fy,
            self.current_cam.cx,
            self.current_cam.cy,
        )
        self.extrinsic_np = make_extrinsic(
            self.current_cam.height,
            self.current_cam.pitch_deg,
            self.current_cam.roll_deg,
        )
        config.DEPTH_MAX = self.current_cam.depth_max
        config.BEV_Z_RANGE = (config.DEPTH_MIN, self.current_cam.depth_max)
        config.BEV_X_RANGE = (-self.current_cam.bev_x_half, self.current_cam.bev_x_half)
        config.ROBOT_RADIUS_M = self.current_cam.robot_radius
        if self.realsense and self.realsense.running:
            if getattr(self.realsense, "intrinsics", None) is not None:
                intr = self.realsense.intrinsics
                self.intrinsic_np = scale_intrinsic(
                    make_intrinsic(intr.fx, intr.fy, intr.ppx, intr.ppy),
                    (intr.width, intr.height),
                    (config.IMG_W, config.IMG_H),
                )
                for name, row, col in (
                    ("fx", 0, 0),
                    ("fy", 1, 1),
                    ("cx", 0, 2),
                    ("cy", 1, 2),
                ):
                    value = getattr(self.current_cam, name)
                    if value is not None:
                        self.intrinsic_np[row, col] = value
            self.realsense.depth_max = self.current_cam.depth_max

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        if self.mode not in {"demo", "realsense", "full"}:
            raise ValueError("BEV_MODE must be demo, realsense, or full")
        self.rebuild_camera()
        if self.mode == "demo":
            logger.info("Offline demo ready; no camera or model downloads")
            try:
                yield
            finally:
                self.close()
            return
        if self.mode == "full":
            from backend.models.depth import DepthEstimator

            self.depth_estimator = DepthEstimator()
            self.depth_estimator.warmup()
        try:
            self.detector = (
                ObjectDetector(conf_thresh=0.35) if self.mode == "full" else None
            )
        except Exception as e:
            self.detector = None
            logger.warning(f"Detector not available ({e}). pip install ultralytics")
        connected = []
        try:
            connected = list_connected_serials()
        except Exception:
            pass
        if len(connected) >= 2:
            try:
                self.multi_rs = MultiRealSenseCapture(
                    depth_min=config.DEPTH_MIN, depth_max=self.current_cam.depth_max
                )
                cam_info = self.multi_rs.start()
                logger.info(
                    f"Multi-camera rig online: {self.multi_rs.n_cameras} camera(s)"
                )
                center = cam_info[self.multi_rs.n_cameras // 2]
                self.intrinsic_np = make_intrinsic(
                    fx=center["fx"], fy=center["fy"], cx=center["cx"], cy=center["cy"]
                )
                ci = self.multi_rs.cams[self.multi_rs.n_cameras // 2]["intr"]
                self.intrinsic_np = scale_intrinsic(
                    self.intrinsic_np,
                    (ci.width, ci.height),
                    (config.IMG_W, config.IMG_H),
                )
            except Exception as e:
                self.multi_rs = None
                logger.warning(
                    f"Multi-camera rig could not start ({e}), trying single camera"
                )
        if self.multi_rs is None and connected:
            try:
                self.realsense = RealSenseCapture(
                    width=640,
                    height=480,
                    fps=30,
                    depth_min=config.DEPTH_MIN,
                    depth_max=self.current_cam.depth_max,
                )
                intr = self.realsense.start()
                self.intrinsic_np = scale_intrinsic(
                    make_intrinsic(fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy),
                    (intr.width, intr.height),
                    (config.IMG_W, config.IMG_H),
                )
                logger.info("RealSense D435i online (single) — using hardware depth")
            except Exception as e:
                self.realsense = None
                logger.warning(
                    f"RealSense not available ({e}) — Depth Anything fallback"
                )
        elif not connected:
            logger.warning("No RealSense cameras detected — Depth Anything fallback")
        try:
            yield
        finally:
            self.close()

    def close(self):
        """Release background workers and camera pipelines."""
        self._yolo_executor.shutdown(wait=True, cancel_futures=True)
        if self.multi_rs:
            self.multi_rs.stop()
        if self.realsense:
            self.realsense.stop()

    def _run_bev_pipeline(self, img_np, metric_depth, closeness):
        if self.current_cam.depth_scale != 1.0:
            metric_depth = metric_depth * self.current_cam.depth_scale
        depth_rgb = turbo_colormap(closeness)
        bev = self.bev_projector.project(
            img_np, metric_depth, self.intrinsic_np, self.extrinsic_np
        )
        fov = compute_fov_mask(bev)
        obstacle, free, unknown = classify_freespace(bev, fov)
        inflated = inflate_obstacles(obstacle)
        self.latest_bev = bev
        self.latest_obstacle = obstacle
        self.latest_inflated = inflated
        self.latest_fov = fov
        self.latest_free = free
        color_img = render_color_bev(bev, show_grid=self.current_cam.show_grid)
        fs_img = render_freespace(
            bev,
            obstacle,
            free,
            unknown,
            fov,
            inflated,
            show_grid=self.current_cam.show_grid,
        )
        return (color_img, fs_img, depth_rgb)

    def _proximity_alert(self, depth_m):
        """Return nearest obstacle distance and severity level."""
        valid = depth_m[
            (depth_m > config.DEPTH_MIN + 0.05) & (depth_m < config.DEPTH_MAX)
        ]
        if len(valid) == 0:
            return (None, "none")
        nearest = float(np.percentile(valid, 2))
        if nearest < 0.5:
            level = "critical"
        elif nearest < 1.0:
            level = "warning"
        else:
            level = "safe"
        return (round(nearest, 2), level)

    def run_inference(self, pil_img: Image.Image, run_detection: bool = True):
        t0 = time.perf_counter()
        img = pil_img.convert("RGB").resize((config.IMG_W, config.IMG_H), Image.LANCZOS)
        img_np = np.array(img)
        if self.depth_estimator is None:
            raise RuntimeError("Image depth estimation requires BEV_MODE=full")
        metric_depth, closeness = self.depth_estimator.estimate(img)
        color_img, fs_img, depth_rgb = self._run_bev_pipeline(
            img_np, metric_depth, closeness
        )
        detections = []
        annotated_img = img_np
        if self.detector and run_detection:
            detections, annotated_img = self.detector.detect(img_np, metric_depth)
        nearest, alert_level = self._proximity_alert(metric_depth)
        elapsed = time.perf_counter() - t0
        return {
            "bev_image": encode_image(color_img),
            "bev_freespace": encode_image(fs_img),
            "depth_image": encode_image(depth_rgb, (config.IMG_W, config.IMG_H)),
            "annotated_image": encode_image(annotated_img),
            "detections": detections,
            "proximity": {"nearest_m": nearest, "level": alert_level},
            "imu": None,
            "inference_ms": round(elapsed * 1000, 1),
            "device": str(config.DEVICE),
            "depth_source": "depth_anything_v2",
            "bev_grid": {
                "x_range": list(config.BEV_X_RANGE),
                "z_range": list(config.BEV_Z_RANGE),
                "size": list(config.BEV_SIZE),
            },
        }

    def run_realsense_inference(self):
        if not self.realsense or not self.realsense.running:
            raise RuntimeError("RealSense not connected")
        t0 = time.perf_counter()
        color_np, metric_depth, closeness, imu_data, valid_ratio = (
            self.realsense.capture()
        )
        color_pil = Image.fromarray(color_np).resize(
            (config.IMG_W, config.IMG_H), Image.LANCZOS
        )
        img_np = np.array(color_pil)
        scale_y = config.IMG_H / metric_depth.shape[0]
        scale_x = config.IMG_W / metric_depth.shape[1]
        depth_resized = ndim_zoom(metric_depth, (scale_y, scale_x), order=0)
        closeness_resized = ndim_zoom(closeness, (scale_y, scale_x), order=1)
        color_img, fs_img, depth_rgb = self._run_bev_pipeline(
            img_np, depth_resized, closeness_resized
        )
        detections = []
        annotated_img = img_np
        if self.detector:
            detections, annotated_img = self.detector.detect(img_np, depth_resized)
        nearest, alert_level = self._proximity_alert(depth_resized)
        elapsed = time.perf_counter() - t0
        return {
            "bev_image": encode_image(color_img),
            "bev_freespace": encode_image(fs_img),
            "depth_image": encode_image(depth_rgb, (config.IMG_W, config.IMG_H)),
            "annotated_image": encode_image(annotated_img),
            "color_image": encode_image(img_np),
            "detections": detections,
            "proximity": {"nearest_m": nearest, "level": alert_level},
            "imu": imu_data,
            "valid_ratio": round(valid_ratio, 3),
            "inference_ms": round(elapsed * 1000, 1),
            "device": "realsense_d435i",
            "depth_source": "realsense_hardware",
            "bev_grid": {
                "x_range": list(config.BEV_X_RANGE),
                "z_range": list(config.BEV_Z_RANGE),
                "size": list(config.BEV_SIZE),
            },
        }

    def run_multi_inference(self):
        """Capture all cameras in parallel, fuse into one BEV."""
        t0 = time.perf_counter()
        from backend.processing.panorama import render_panorama
        from backend.processing.surround import render_surround_bev

        captures = self.multi_rs.capture_all()
        if not captures:
            raise RuntimeError("All cameras failed to capture")
        bev_colors = []

        cam_frames = []
        all_dets = []
        center_imu = None
        min_dist = float("inf")
        import cv2

        all_bevs = []
        all_fovs = []

        pano_frames = []
        surround_frames = []
        bev_dets_raw = []
        for cam_dict, color_np, depth_m, closeness, imu, valid_ratio in captures:
            E = make_extrinsic(
                self.current_cam.height,
                self.current_cam.pitch_deg,
                roll_deg=0.0,
                yaw_deg=cam_dict["yaw_deg"],
            )
            intr = cam_dict.get("intr")
            K = make_intrinsic(
                fx=intr.fx if intr else None,
                fy=intr.fy if intr else None,
                cx=intr.ppx if intr else None,
                cy=intr.ppy if intr else None,
            )
            if intr:
                K = scale_intrinsic(
                    K, (intr.width, intr.height), (config.IMG_W, config.IMG_H)
                )
            color_pil = Image.fromarray(color_np).resize(
                (config.IMG_W, config.IMG_H), Image.LANCZOS
            )
            img_np = np.array(color_pil)
            depth_r = cv2.resize(
                depth_m, (config.IMG_W, config.IMG_H), interpolation=cv2.INTER_NEAREST
            )
            close_r = cv2.resize(
                closeness, (config.IMG_W, config.IMG_H), interpolation=cv2.INTER_NEAREST
            )
            if self.current_cam.depth_scale != 1.0:
                depth_r = depth_r * self.current_cam.depth_scale
            depth_rgb = turbo_colormap(close_r)
            bev = self.bev_projector.project(img_np, depth_r, K, E)
            fov = compute_fov_mask(bev, yaw_deg=cam_dict["yaw_deg"])
            all_bevs.append(bev)
            all_fovs.append(fov)
            bev_colors.append(render_color_bev(bev, show_grid=False))
            valid = depth_r[
                (depth_r > config.DEPTH_MIN + 0.05) & (depth_r < config.DEPTH_MAX)
            ]
            if len(valid):
                min_dist = min(min_dist, float(np.percentile(valid, 2)))
            if cam_dict["label"] == "CENTER":
                center_imu = imu

            pano_frames.append(
                (img_np.copy(), cam_dict.get("intr"), cam_dict["yaw_deg"])
            )
            surround_frames.append(
                (img_np.copy(), K.copy(), E.copy(), cam_dict["yaw_deg"])
            )
            run_det = self.current_cam.detection_mode != "off" and (
                self.current_cam.detection_mode == "all"
                or cam_dict["label"] == "CENTER"
            )
            if run_det:
                with self._yolo_lock:
                    dets, annotated = self._yolo_cache.get(
                        cam_dict["label"], ([], img_np)
                    )
                self._yolo_executor.submit(
                    self._run_yolo_bg, cam_dict["label"], img_np.copy(), depth_r.copy()
                )
                for det in dets:
                    if det.get("distance") is None:
                        det["distance"] = round(self.current_cam.height * 1.5, 1)
                all_dets.extend(dets)
                if self.current_cam.bev_detect:
                    for d in dets:
                        bev_dets_raw.append((d, K.copy(), E.copy()))
            else:
                dets, annotated = ([], img_np)
            cam_frames.append(
                {
                    "label": cam_dict["label"],
                    "yaw_deg": cam_dict["yaw_deg"],
                    "annotated_image": encode_image(annotated),
                    "depth_image": encode_image(
                        depth_rgb, (config.IMG_W, config.IMG_H)
                    ),
                    "detections": dets,
                    "valid_ratio": round(valid_ratio, 3),
                }
            )
        fused_bev = fuse_results(all_bevs)
        merged_color = fused_bev.color_image.copy()
        logger.debug(
            f"Multi-cam: {len(captures)} captures, {len(pano_frames)} pano frames collected"
        )
        bev_clean_b64 = encode_image(merged_color.copy())
        if self.current_cam.bev_detect and bev_dets_raw:
            draw_bev_detections(
                merged_color,
                bev_dets_raw,
                x_range=config.BEV_X_RANGE,
                z_range=config.BEV_Z_RANGE,
                bev_size=config.BEV_SIZE,
            )
        pano_b64 = None
        try:
            pano_np = render_panorama(
                pano_frames,
                cap_w=424,
                cap_h=240,
                out_w=config.IMG_W,
                out_h=config.IMG_H,
            )
            if pano_np is not None:
                pano_b64 = encode_image(pano_np)
                logger.debug(
                    f"Panorama rendered: {pano_np.shape}, b64_len={(len(pano_b64) if pano_b64 else 0)}"
                )
            else:
                logger.warning(
                    f"render_panorama returned None — pano_frames={len(pano_frames)}"
                )
        except Exception as e:
            logger.warning(f"Panorama render failed: {e}", exc_info=True)
        surround_b64 = None
        try:
            bev_h, bev_w = config.BEV_SIZE
            surround_np = render_surround_bev(
                surround_frames,
                bev_h=bev_h,
                bev_w=bev_w,
                x_min=config.BEV_X_RANGE[0],
                x_max=config.BEV_X_RANGE[1],
                z_min=config.DEPTH_MIN,
                z_max=self.current_cam.depth_max,
                show_grid=self.current_cam.show_grid,
            )
            if surround_np is not None:
                surround_b64 = encode_image(surround_np)
        except Exception as e:
            logger.warning(f"Surround BEV render failed: {e}", exc_info=True)
        combined_fov = all_fovs[0].copy()
        combined_obs = np.zeros_like(combined_fov)
        combined_free = np.zeros_like(combined_fov)
        for bev, fov in zip(all_bevs, all_fovs):
            obs, free, _ = classify_freespace(bev, fov)
            combined_fov |= fov
            combined_obs |= obs
            combined_free |= free
        combined_free &= ~combined_obs
        combined_unknown = combined_fov & ~combined_obs & ~combined_free
        combined_inflated = inflate_obstacles(combined_obs)
        centre_bev = fused_bev
        centre_bev.color_image = merged_color
        merged_free = render_freespace(
            centre_bev,
            combined_obs,
            combined_free,
            combined_unknown,
            combined_fov,
            combined_inflated,
            show_grid=False,
        )
        self.latest_bev = centre_bev
        self.latest_obstacle = combined_obs
        self.latest_inflated = combined_inflated
        self.latest_fov = combined_fov
        self.latest_free = combined_free
        nearest = round(min_dist, 2) if min_dist < float("inf") else None
        level = (
            "critical"
            if nearest and nearest < 0.5
            else "warning"
            if nearest and nearest < 1.0
            else "safe"
        )
        elapsed = time.perf_counter() - t0
        return {
            "bev_image": encode_image(merged_color),
            "bev_clean": bev_clean_b64,
            "bev_freespace": encode_image(merged_free),
            "panorama_image": pano_b64,
            "surround_image": surround_b64,
            "camera_frames": cam_frames,
            "detections": all_dets,
            "proximity": {"nearest_m": nearest, "level": level},
            "imu": center_imu,
            "n_cameras": len(captures),
            "inference_ms": round(elapsed * 1000, 1),
            "depth_source": "realsense_multi",
            "bev_grid": {
                "x_range": list(config.BEV_X_RANGE),
                "z_range": list(config.BEV_Z_RANGE),
                "size": list(config.BEV_SIZE),
            },
        }
