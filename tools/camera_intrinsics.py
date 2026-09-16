"""Print color-stream intrinsics for a connected RealSense camera."""


def main():
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    profile = pipeline.start(config)
    try:
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        print(f"Resolution: {intr.width} x {intr.height}")
        print(f"Focal length: fx={intr.fx:.4f}, fy={intr.fy:.4f}")
        print(f"Principal point: cx={intr.ppx:.4f}, cy={intr.ppy:.4f}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
