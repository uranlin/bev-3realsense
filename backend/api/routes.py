"""HTTP endpoints bound to a navigation runtime."""

from __future__ import annotations

import base64
import io
import traceback

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from backend import config
from backend.api.schemas import (
    Base64Request,
    CameraConfig,
    DepthModelRequest,
    MultiCamConfig,
    PlanRequest,
    SourceRequest,
)
from backend.models.realsense_multi import list_connected_serials
from backend.processing import (
    classify_freespace,
    compute_costmap,
    inflate_obstacles,
    path_grid_to_world,
    plan_path,
    render_color_bev,
    render_freespace,
    world_to_grid,
)
from backend.services.runtime import DEPTH_MODELS, logger
from backend.utils import encode_image, make_intrinsic
from backend.utils.camera import scale_intrinsic


def create_router(runtime):
    router = APIRouter()

    @router.post("/infer_base64")
    async def infer_base64(req: Base64Request):
        try:
            b64 = req.image.split(",")[-1]
            pil_img = Image.open(io.BytesIO(base64.b64decode(b64)))
            pil_img.load()
        except Exception as e:
            raise HTTPException(400, detail=f"Invalid image: {e}")
        try:
            return JSONResponse(runtime.run_inference(pil_img))
        except Exception as e:
            logger.error(traceback.format_exc())
            raise HTTPException(500, detail=f"Inference failed: {e}")

    @router.post("/infer")
    async def infer_upload(file: UploadFile = File(...)):
        try:
            pil_img = Image.open(io.BytesIO(await file.read()))
            pil_img.load()
        except Exception as e:
            raise HTTPException(400, detail=f"Invalid image: {e}")
        try:
            return JSONResponse(runtime.run_inference(pil_img))
        except Exception as e:
            logger.error(traceback.format_exc())
            raise HTTPException(500, detail=f"Inference failed: {e}")

    @router.post("/infer_realsense")
    async def infer_realsense_endpoint():
        if not runtime.realsense or not runtime.realsense.running:
            raise HTTPException(400, detail="RealSense not connected.")
        try:
            return JSONResponse(runtime.run_realsense_inference())
        except Exception as e:
            logger.error(traceback.format_exc())
            raise HTTPException(500, detail=f"RealSense inference failed: {e}")

    @router.post("/plan")
    async def plan(req: PlanRequest):
        if runtime.latest_bev is None or runtime.latest_inflated is None:
            raise HTTPException(400, detail="Run inference first.")
        try:
            start_rc = (config.BEV_SIZE[0] - 5, config.BEV_SIZE[1] // 2)
            goal_rc = world_to_grid(req.goal_x, req.goal_z)
            inflated = runtime.latest_inflated
            if (
                req.robot_radius is not None
                and req.robot_radius != config.ROBOT_RADIUS_M
            ):
                inflated = inflate_obstacles(runtime.latest_obstacle, req.robot_radius)
            costmap = compute_costmap(inflated, runtime.latest_free)
            path_cells = plan_path(
                inflated, runtime.latest_free, start_rc, goal_rc, costmap
            )
            if path_cells is None:
                return JSONResponse({"success": False, "message": "No path found."})
            path_world = path_grid_to_world(path_cells)
            total_dist = sum(
                (
                    (
                        (path_world[i][0] - path_world[i - 1][0]) ** 2
                        + (path_world[i][1] - path_world[i - 1][1]) ** 2
                    )
                    ** 0.5
                    for i in range(1, len(path_world))
                )
            )
            free = runtime.latest_free
            unknown = runtime.latest_fov & ~runtime.latest_obstacle & ~free
            fs_img = render_freespace(
                runtime.latest_bev,
                runtime.latest_obstacle,
                free,
                unknown,
                runtime.latest_fov,
                inflated,
                path_cells,
                show_grid=runtime.current_cam.show_grid,
            )
            color_img = render_color_bev(
                runtime.latest_bev, path_world, show_grid=runtime.current_cam.show_grid
            )
            return JSONResponse(
                {
                    "success": True,
                    "path_world": path_world,
                    "path_length_m": round(total_dist, 2),
                    "n_waypoints": len(path_world),
                    "bev_image": encode_image(color_img),
                    "bev_freespace": encode_image(fs_img),
                }
            )
        except ValueError as e:
            raise HTTPException(422, detail=str(e))
        except Exception as e:
            logger.error(traceback.format_exc())
            raise HTTPException(500, detail=f"Planning failed: {e}")

    @router.get("/config")
    async def get_config():
        single_ok = runtime.realsense is not None and runtime.realsense.running
        multi_ok = runtime.multi_rs is not None and runtime.multi_rs.running
        return {
            "camera": runtime.current_cam.model_dump(),
            "img_size": [config.IMG_W, config.IMG_H],
            "bev_size": list(config.BEV_SIZE),
            "depth_min": config.DEPTH_MIN,
            "depth_max": runtime.current_cam.depth_max,
            "depth_range": f"{config.DEPTH_MIN}-{runtime.current_cam.depth_max}m",
            "robot_radius": config.ROBOT_RADIUS_M,
            "device": str(config.DEVICE),
            "realsense_available": single_ok or multi_ok,
            "multi_rig_active": multi_ok,
            "n_cameras": runtime.multi_rs.n_cameras
            if multi_ok
            else 1
            if single_ok
            else 0,
            "camera_info": runtime.multi_rs.camera_info() if multi_ok else [],
            "detector_available": runtime.detector is not None,
            "depth_model": runtime.current_depth_model_key,
            "bev_grid": {
                "x_range": list(config.BEV_X_RANGE),
                "z_range": list(config.BEV_Z_RANGE),
                "size": list(config.BEV_SIZE),
            },
        }

    @router.post("/config")
    async def update_config(cfg: CameraConfig):
        runtime.latest_bev = runtime.latest_free = runtime.latest_obstacle = (
            runtime.latest_inflated
        ) = runtime.latest_fov = None
        runtime.current_cam = cfg
        runtime.rebuild_camera()
        return {
            "status": "ok",
            "depth_range": f"{config.DEPTH_MIN}-{cfg.depth_max}m",
            "bev_z_range": list(config.BEV_Z_RANGE),
            "bev_x_range": list(config.BEV_X_RANGE),
        }

    @router.post("/depth_model")
    async def switch_depth_model(req: DepthModelRequest):
        """Hot-swap Depth Anything V2 between indoor and outdoor metric models."""
        if runtime.mode != "full":
            raise HTTPException(409, "Model loading requires BEV_MODE=full")
        from backend.models.depth import DepthEstimator

        key = req.model.lower()
        if key not in DEPTH_MODELS:
            return {"status": "error", "message": f"Unknown model key '{key}'"}
        if key == runtime.current_depth_model_key:
            return {"status": "ok", "model": key, "message": "Already loaded"}
        try:
            model_name = DEPTH_MODELS[key]
            logger.info(f"Switching depth model → {key} ({model_name})")
            import gc

            import torch

            del runtime.depth_estimator
            runtime.depth_estimator = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import backend.config as _cfg

            _cfg.DEPTH_MODEL = model_name
            runtime.depth_estimator = DepthEstimator()
            runtime.depth_estimator.warmup()
            runtime.current_depth_model_key = key
            logger.info(
                f"Depth model switched to {key} — {runtime.depth_estimator.n_params:.1f}M params"
            )
            return {
                "status": "ok",
                "model": key,
                "params_m": round(runtime.depth_estimator.n_params, 1),
            }
        except Exception as e:
            logger.error(f"Depth model switch failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    @router.post("/source")
    async def switch_source(req: SourceRequest):
        """Hot-swap between RealSense hardware depth and webcam + Depth Anything V2."""
        if req.source == "realsense":
            if not runtime.realsense:
                raise HTTPException(
                    400, detail="RealSense not connected. Check USB and restart server."
                )
            if not runtime.realsense.running:
                try:
                    intr = runtime.realsense.start()
                    runtime.intrinsic_np = scale_intrinsic(
                        make_intrinsic(
                            fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy
                        ),
                        (intr.width, intr.height),
                        (config.IMG_W, config.IMG_H),
                    )
                    logger.info("RealSense restarted for source switch.")
                except Exception as e:
                    raise HTTPException(500, detail=f"Failed to start RealSense: {e}")
            return {"status": "ok", "source": "realsense", "realsense_available": True}
        elif req.source == "webcam":
            if runtime.realsense and runtime.realsense.running:
                runtime.realsense.stop()
                logger.info("RealSense stopped — releasing USB for webcam mode.")
                runtime.rebuild_camera()
            return {"status": "ok", "source": "webcam", "realsense_available": False}
        raise HTTPException(400, detail=f"Unknown source: {req.source}")

    @router.get("/cameras")
    async def list_cameras():
        """List all detected RealSense cameras and their current status."""
        try:
            connected = list_connected_serials()
        except Exception:
            connected = []
        return {
            "connected_serials": connected,
            "n_connected": len(connected),
            "multi_rig_active": runtime.multi_rs is not None
            and runtime.multi_rs.running,
            "single_cam_active": runtime.realsense is not None
            and runtime.realsense.running,
            "cameras": runtime.multi_rs.camera_info() if runtime.multi_rs else [],
        }

    @router.post("/multi_cam_config")
    async def update_multi_cam_config(cfg: MultiCamConfig):
        """Update yaw angles for the 3-camera rig without restarting pipelines."""
        runtime.multi_cam_cfg = cfg
        if runtime.multi_rs:
            yaws = [cfg.yaw_left, cfg.yaw_center, cfg.yaw_right]
            for i in range(min(runtime.multi_rs.n_cameras, 3)):
                runtime.multi_rs.set_yaw(i, yaws[i])
        return {"status": "ok", "config": cfg.model_dump()}

    @router.post("/infer_multi")
    async def infer_multi_endpoint():
        """Capture from all cameras and return a fused BEV + per-camera frames."""
        if not runtime.multi_rs or not runtime.multi_rs.running:
            raise HTTPException(
                400,
                detail="Multi-camera rig not active. Connect ≥2 RealSense cameras and restart the server.",
            )
        try:
            return JSONResponse(runtime.run_multi_inference())
        except Exception as e:
            logger.error(traceback.format_exc())
            raise HTTPException(500, detail=f"Multi-camera inference failed: {e}")

    @router.get("/health")
    async def health():
        return {
            "status": "ok",
            "mode": runtime.mode,
            "depth_model_loaded": runtime.depth_estimator is not None,
        }

    @router.post("/demo")
    async def demo():
        if runtime.mode != "demo":
            raise HTTPException(409, "Restart with BEV_MODE=demo to use synthetic data")
        from backend.demo import make_demo_scene

        runtime.latest_bev, runtime.latest_fov = make_demo_scene()
        runtime.latest_obstacle, runtime.latest_free, unknown = classify_freespace(
            runtime.latest_bev, runtime.latest_fov
        )
        runtime.latest_inflated = inflate_obstacles(runtime.latest_obstacle)
        return {
            "bev_image": encode_image(runtime.latest_bev.color_image),
            "bev_freespace": encode_image(
                render_freespace(
                    runtime.latest_bev,
                    runtime.latest_obstacle,
                    runtime.latest_free,
                    unknown,
                    runtime.latest_fov,
                    runtime.latest_inflated,
                )
            ),
            "depth_source": "synthetic_bev",
            "n_cameras": 3,
        }

    return router
