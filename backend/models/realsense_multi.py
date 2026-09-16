"""
Multi-camera RealSense D435i manager.

Discovers up to 3 connected cameras, assigns LEFT / CENTER / RIGHT roles,
and captures frames in parallel using ThreadPoolExecutor.

Fault-tolerant: if a camera fails to start, it is skipped and the rig
continues with the remaining cameras.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

logger = logging.getLogger("bev-nav")

_YAW_DEFAULTS = {1: [0.0], 2: [-30.0, 30.0], 3: [-45.0, 0.0, 45.0]}
_LABELS = {1: ["CENTER"], 2: ["LEFT", "RIGHT"], 3: ["LEFT", "CENTER", "RIGHT"]}

# Resolution fallback ladder — tried in order until one works
_RES_LADDER = [
    (424, 240, 15),
    (320, 180, 10),
    (320, 240, 6),
]


def list_connected_serials():
    """Return sorted serial numbers of all connected RealSense devices."""
    import pyrealsense2 as rs

    ctx = rs.context()
    return sorted(dev.get_info(rs.camera_info.serial_number) for dev in ctx.devices)


class _SingleCamera:
    """Thin wrapper around one RealSense pipeline opened by serial."""

    def __init__(
        self, serial, width=424, height=240, fps=15, depth_min=0.2, depth_max=5.0
    ):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_min = depth_min
        self.depth_max = depth_max
        self._pipeline = None
        self._align = None
        self._scale = 1.0
        self._filters = []
        self._has_imu = False
        self.running = False

    def start(self):

        import pyrealsense2 as rs

        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
        )
        cfg.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        try:
            cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
            cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
            self._has_imu = True
        except Exception:
            self._has_imu = False

        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(cfg)

        sensor = profile.get_device().first_depth_sensor()
        self._scale = sensor.get_depth_scale()

        # Max IR laser power → better depth on blank walls/floors
        try:
            sensor.set_option(rs.option.emitter_enabled, 1)
            max_power = sensor.get_option_range(rs.option.laser_power).max
            sensor.set_option(rs.option.laser_power, max_power)
        except Exception:
            pass
        self._align = rs.align(rs.stream.color)

        sp = rs.spatial_filter()
        sp.set_option(rs.option.filter_magnitude, 5)
        sp.set_option(rs.option.filter_smooth_alpha, 0.75)
        sp.set_option(rs.option.filter_smooth_delta, 1.0)
        tp = rs.temporal_filter()
        tp.set_option(rs.option.filter_smooth_alpha, 0.4)
        tp.set_option(rs.option.filter_smooth_delta, 20.0)
        hf = rs.hole_filling_filter()
        hf.set_option(rs.option.holes_fill, 1)
        self._filters = [sp, tp, hf]
        self.running = True

        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        return intr

    def start_with_fallback(self):
        """Try progressively lower resolutions until one works."""
        last_err = None
        for w, h, fps in _RES_LADDER:
            self.width, self.height, self.fps = w, h, fps
            try:
                return self.start()
            except Exception as e:
                last_err = e
                logger.warning(
                    f"    {self.serial} @ {w}×{h}@{fps}fps failed: {e} — trying lower"
                )
                if self._pipeline:
                    try:
                        self._pipeline.stop()
                    except Exception:
                        pass
                    self._pipeline = None
                self.running = False
        raise RuntimeError(f"All resolutions failed for {self.serial}: {last_err}")

    def stop(self):
        if self._pipeline and self.running:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self.running = False

    def capture(self):
        import math

        import pyrealsense2 as rs

        if not self.running:
            raise RuntimeError(f"Camera {self.serial} not running")

        for _ in range(3):
            try:
                self._pipeline.wait_for_frames(timeout_ms=50)
            except Exception:
                break

        frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        aligned = self._align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError(f"Camera {self.serial}: bad frame")

        for f in self._filters:
            depth_frame = f.process(depth_frame)

        color_np = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * self._scale

        valid_mask = (
            np.isfinite(depth_m)
            & (depth_m >= self.depth_min)
            & (depth_m <= self.depth_max)
        )
        valid_ratio = float(valid_mask.mean())
        depth_m = np.where(valid_mask, depth_m, 0.0)

        d_lo = float(np.percentile(depth_m, 2))
        d_hi = float(np.percentile(depth_m, 98))
        closeness = 1.0 - (depth_m - d_lo) / max(d_hi - d_lo, 0.1)
        closeness = np.clip(closeness, 0.0, 1.0)

        imu_data = None
        if self._has_imu:
            try:
                af = frames.first_or_default(rs.stream.accel)
                if af and af.is_motion_frame():
                    a = af.as_motion_frame().get_motion_data()
                    roll = math.degrees(math.atan2(a.y, a.z))
                    pitch = math.degrees(math.atan2(-a.x, math.sqrt(a.y**2 + a.z**2)))
                    imu_data = {
                        "roll": round(roll, 1),
                        "pitch": round(pitch, 1),
                        "accel": {
                            "x": round(a.x, 3),
                            "y": round(a.y, 3),
                            "z": round(a.z, 3),
                        },
                    }
            except Exception:
                pass

        return color_np, depth_m, closeness, imu_data, valid_ratio


class MultiRealSenseCapture:
    """
    Manages 1–3 RealSense cameras with parallel frame capture.
    Fault-tolerant: skips cameras that fail to start.
    """

    def __init__(self, serials=None, yaw_degs=None, depth_min=0.2, depth_max=5.0):
        connected = list_connected_serials()
        if not connected:
            raise RuntimeError("No RealSense cameras connected")

        self._all_serials = (serials or connected)[:3]
        n_requested = len(self._all_serials)

        # Assign yaw angles based on requested count
        requested_yaws = (
            yaw_degs or _YAW_DEFAULTS.get(n_requested, [0.0] * n_requested)
        )[:n_requested]
        requested_labels = _LABELS.get(
            n_requested, [f"CAM{i}" for i in range(n_requested)]
        )

        # Build camera slots (may be reduced after start())
        self._slots = []
        for i, serial in enumerate(self._all_serials):
            self._slots.append(
                {
                    "cam": _SingleCamera(
                        serial, depth_min=depth_min, depth_max=depth_max
                    ),
                    "serial": serial,
                    "yaw_deg": requested_yaws[i],
                    "label": requested_labels[i],
                    "intr": None,
                }
            )

        self.cams = []  # populated after start() with only successful cameras
        self.running = False

    def start(self):
        """
        Start all cameras with resolution fallback.
        Skips cameras that fail completely — continues with the rest.
        Returns list of camera info dicts for cameras that started.
        """
        info = []
        for slot in self._slots:
            try:
                intr = slot["cam"].start_with_fallback()
                slot["intr"] = intr
                self.cams.append(slot)
                logger.info(
                    f"  {slot['label']} ({slot['serial']}): "
                    f"fx={intr.fx:.1f} @ "
                    f"{slot['cam'].width}×{slot['cam'].height}@{slot['cam'].fps}fps "
                    f"| yaw={slot['yaw_deg']}°"
                )
                info.append(
                    {
                        "label": slot["label"],
                        "serial": slot["serial"],
                        "yaw_deg": slot["yaw_deg"],
                        "fx": intr.fx,
                        "fy": intr.fy,
                        "cx": intr.ppx,
                        "cy": intr.ppy,
                    }
                )
            except Exception as e:
                logger.warning(f"  {slot['label']} ({slot['serial']}) skipped — {e}")

        if not self.cams:
            raise RuntimeError("No cameras could start — check USB connections")

        # Reassign labels for the cameras that actually started
        n = len(self.cams)
        new_labels = _LABELS.get(n, [f"CAM{i}" for i in range(n)])
        for i, cam in enumerate(self.cams):
            cam["label"] = new_labels[i]
            if i < len(info):
                info[i]["label"] = new_labels[i]

        self._serials = [c["serial"] for c in self.cams]
        self.running = True

        logger.info(
            f"Multi-camera rig online: {n}/{len(self._slots)} camera(s) started"
        )
        return info

    def stop(self):
        for c in self.cams:
            try:
                c["cam"].stop()
            except Exception:
                pass
        self.running = False

    def capture_all(self):
        """Parallel capture from all active cameras."""
        bucket = {}

        with ThreadPoolExecutor(max_workers=len(self.cams)) as ex:
            futs = {ex.submit(c["cam"].capture): c for c in self.cams}
            for fut in as_completed(futs):
                c = futs[fut]
                try:
                    result = fut.result()
                    bucket[c["serial"]] = (c,) + result
                except Exception as e:
                    logger.warning(f"Capture error {c['label']}: {e}")

        return [bucket[s] for s in self._serials if s in bucket]

    @property
    def n_cameras(self):
        return len(self.cams)

    def set_yaw(self, index, yaw_deg):
        if 0 <= index < len(self.cams):
            self.cams[index]["yaw_deg"] = yaw_deg

    def camera_info(self):
        return [
            {
                "label": c["label"],
                "serial": c["serial"],
                "yaw_deg": c["yaw_deg"],
                "running": c["cam"].running,
                "width": c["cam"].width,
                "height": c["cam"].height,
                "fps": c["cam"].fps,
            }
            for c in self.cams
        ]
