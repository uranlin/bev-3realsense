"""
RealSense D435i capture module — depth + color + IMU.

Enhancements over the basic version:
  • Intel SDK spatial / temporal / hole-filling filters (replaces manual scipy fill)
  • IMU streams (accelerometer + gyroscope) for roll/pitch estimation
"""

import logging
import math

import numpy as np

logger = logging.getLogger("bev-nav")


class RealSenseCapture:
    """RealSense D435i pipeline with aligned RGB-D frames."""

    def __init__(self, width=640, height=480, fps=30, depth_min=0.2, depth_max=5.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_min = depth_min
        self.depth_max = depth_max

        self._pipeline = None
        self._align = None
        self._scale = 1.0
        self._filters = []  # SDK post-processing chain
        self.running = False

    def start(self):
        import pyrealsense2 as rs

        cfg = rs.config()
        cfg.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
        )
        cfg.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )

        # IMU streams (D435i only — D435 without 'i' has no IMU)
        try:
            cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
            cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
            self._has_imu = True
        except Exception:
            self._has_imu = False
            logger.warning("IMU streams not available on this device.")

        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(cfg)

        sensor = profile.get_device().first_depth_sensor()
        self._scale = sensor.get_depth_scale()

        self._align = rs.align(rs.stream.color)

        # 1. Decimation: halve resolution for speed (optional — skip if quality > speed)
        # dec = rs.decimation_filter(); dec.set_option(rs.option.filter_magnitude, 2)
        # self._filters.append(dec)

        # 2. Spatial filter: fills holes using surrounding valid pixels
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 5)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.75)
        spatial.set_option(rs.option.filter_smooth_delta, 1.0)
        self._filters.append(spatial)

        # 3. Temporal filter: reduces flickering by averaging across frames
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        temporal.set_option(rs.option.filter_smooth_delta, 20.0)
        self._filters.append(temporal)

        # 4. Hole filling: fills remaining invalid pixels
        hole_fill = rs.hole_filling_filter()
        hole_fill.set_option(rs.option.holes_fill, 1)  # 1 = farthest from around
        self._filters.append(hole_fill)

        self.running = True

        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        logger.info(
            f"RealSense started | depth_scale={self._scale:.6f} "
            f"| fx={intr.fx:.1f} fy={intr.fy:.1f} "
            f"cx={intr.ppx:.1f} cy={intr.ppy:.1f} "
            f"| IMU={'yes' if self._has_imu else 'no'}"
        )
        self.intrinsics = intr
        return intr

    def stop(self):
        if self._pipeline and self.running:
            self._pipeline.stop()
            self.running = False
            logger.info("RealSense stopped.")

    def capture(self):
        """
        Capture one aligned RGBD frame + IMU data.

        Returns
        -------
        color_np  : (H, W, 3) uint8 RGB
        depth_m   : (H, W)    float32 metric depth in metres (filtered)
        closeness : (H, W)    float32 [0=far, 1=near] for colormap
        imu_data  : dict with keys: roll, pitch, accel, gyro (or None)
        valid_ratio: float — fraction of valid depth pixels
        """
        import pyrealsense2 as rs

        if not self.running:
            raise RuntimeError("RealSense pipeline not started")

        # Drain stale buffered frames (get freshest frame)
        for _ in range(5):
            try:
                self._pipeline.wait_for_frames(timeout_ms=100)
            except Exception:
                break

        frames = self._pipeline.wait_for_frames(timeout_ms=10000)

        imu_data = None
        if self._has_imu:
            try:
                accel_frame = frames.first_or_default(rs.stream.accel)
                gyro_frame = frames.first_or_default(rs.stream.gyro)

                if accel_frame and accel_frame.is_motion_frame():
                    a = accel_frame.as_motion_frame().get_motion_data()
                    g = (
                        gyro_frame.as_motion_frame().get_motion_data()
                        if (gyro_frame and gyro_frame.is_motion_frame())
                        else None
                    )

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
                        "gyro": {
                            "x": round(g.x, 3) if g else 0,
                            "y": round(g.y, 3) if g else 0,
                            "z": round(g.z, 3) if g else 0,
                        }
                        if g
                        else None,
                    }
            except Exception as e:
                logger.debug(f"IMU read error: {e}")

        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to get aligned RGBD frames")

        # Apply SDK filter chain
        for filt in self._filters:
            depth_frame = filt.process(depth_frame)

        color_np = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * self._scale

        # Valid mask (SDK filters handle hole-filling already)
        valid_mask = (
            np.isfinite(depth_m)
            & (depth_m >= self.depth_min)
            & (depth_m <= self.depth_max)
        )
        valid_ratio = float(valid_mask.mean())

        depth_m = np.where(valid_mask, depth_m, 0.0)

        # Colormap: normalise to scene's actual range for full contrast
        d_lo = float(np.percentile(depth_m, 2))
        d_hi = float(np.percentile(depth_m, 98))
        closeness = 1.0 - (depth_m - d_lo) / max(d_hi - d_lo, 0.1)
        closeness = np.clip(closeness, 0.0, 1.0)

        if valid_ratio < 0.05:
            logger.warning(
                f"Poor depth ({valid_ratio:.0%} valid). "
                "Aim at matte surfaces 0.5–4m away."
            )

        return color_np, depth_m, closeness, imu_data, valid_ratio

    @staticmethod
    def get_intrinsics(width=640, height=480):
        import pyrealsense2 as rs

        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
        pipe = rs.pipeline()
        profile = pipe.start(cfg)
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        pipe.stop()
        return intr
