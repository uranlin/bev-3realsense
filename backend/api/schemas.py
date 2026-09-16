"""Validated HTTP request payloads."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class CameraConfig(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    fx: Optional[float] = Field(None, gt=0)
    fy: Optional[float] = Field(None, gt=0)
    cx: Optional[float] = None
    cy: Optional[float] = None
    height: float = Field(1.5, gt=0, le=4)
    pitch_deg: float = -5.0
    roll_deg: float = 0.0
    depth_max: float = Field(5.0, gt=0.3, le=100)
    bev_x_half: float = Field(4.0, gt=0, le=100)
    show_grid: bool = True
    depth_scale: float = Field(1.0, gt=0, le=100)
    robot_radius: float = Field(0.25, ge=0, le=5)
    detection_mode: Literal["off", "center", "all"] = "center"
    bev_detect: bool = False


class SourceRequest(BaseModel):
    source: str


class MultiCamConfig(BaseModel):
    """Per-camera yaw angles for the 3-camera rig."""

    yaw_left: float = -45.0
    yaw_center: float = 0.0
    yaw_right: float = 45.0


class PlanRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    goal_x: float
    goal_z: float
    robot_radius: Optional[float] = Field(None, ge=0, le=5)


class Base64Request(BaseModel):
    image: str


class DepthModelRequest(BaseModel):
    model: str
