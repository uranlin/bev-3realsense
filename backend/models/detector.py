"""
YOLOv8-nano object detector with metric depth distance measurement.

For each detected object, samples the depth map at the bounding box
centre to report metric distance in metres.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger("bev-nav")

# Per-class box colors (cycling palette)
_PALETTE = [
    (255, 59, 48),  # red
    (255, 149, 0),  # orange
    (52, 199, 89),  # green
    (48, 176, 199),  # teal
    (0, 122, 255),  # blue
    (175, 82, 222),  # purple
    (255, 204, 0),  # yellow
    (255, 45, 85),  # pink
]


def _cls_color(name: str):
    return _PALETTE[hash(name) % len(_PALETTE)]


class ObjectDetector:
    """YOLOv8-nano real-time detection with metric depth annotation."""

    def __init__(self, conf_thresh: float = 0.35):
        import torch
        from ultralytics import YOLO

        self.conf = conf_thresh
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(
            str(Path(__file__).resolve().parents[2] / "weights" / "yolov8n.pt")
        )  # ~6 MB — downloads once on first run
        self.model.to(self.device)
        logger.info(
            f"YOLOv8-nano loaded on {self.device} ({len(self.model.names)} classes)"
        )

    def detect(self, img_np: np.ndarray, depth_m: np.ndarray):
        """
        Detect objects and measure distance from depth map.

        Args:
            img_np:  (H, W, 3) uint8 RGB image
            depth_m: (H, W) float32 metric depth in metres

        Returns:
            detections: list of dicts — bbox, class, conf, distance_m
            annotated:  (H, W, 3) uint8 image with boxes + distance labels
        """
        ih, iw = img_np.shape[:2]
        dh, dw = depth_m.shape

        results = self.model(img_np, conf=self.conf, verbose=False)

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = self.model.names[cls_id]
            dist = self._sample_depth(depth_m, x1, y1, x2, y2, iw, ih, dw, dh)

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class": name,
                    "conf": round(conf, 2),
                    "distance": round(dist, 2) if dist > 0 else None,
                }
            )

        annotated = self._draw(img_np, detections)
        return detections, annotated

    @staticmethod
    def _sample_depth(depth_m, x1, y1, x2, y2, iw, ih, dw, dh):
        """Median depth of a patch centred on the bounding box."""
        # Scale box coordinates to depth resolution
        cx = int((x1 + x2) / 2 * dw / iw)
        cy = int((y1 + y2) / 2 * dh / ih)
        pw = max(3, int((x2 - x1) * dw / iw) // 4)
        ph = max(3, int((y2 - y1) * dh / ih) // 4)

        patch = depth_m[
            max(0, cy - ph) : min(dh, cy + ph),
            max(0, cx - pw) : min(dw, cx + pw),
        ]
        valid = patch[(patch > 0.1) & (patch < 98.0)]
        return float(np.median(valid)) if len(valid) > 3 else -1.0

    @staticmethod
    def _draw(img_np: np.ndarray, detections: list) -> np.ndarray:
        pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(pil)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = _cls_color(det["class"])
            light = tuple(min(255, c + 80) for c in color)

            # Bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Label text
            dist_str = f" {det['distance']:.1f}m" if det["distance"] else ""
            label = f"{det['class']}{dist_str}"

            # Label background — clamp to image bounds so it's never clipped
            pad, th = 4, 14
            tw = len(label) * 7
            iw = img_np.shape[1]
            lx1 = max(0, min(x1, iw - tw - pad * 2))  # clamp left edge
            ly1 = max(0, y1 - th - pad * 2)
            lx2 = lx1 + tw + pad * 2
            ly2 = max(0, y1)
            draw.rectangle([lx1, ly1, lx2, ly2], fill=color)
            draw.text((lx1 + pad, ly1 + 1), label, fill=(255, 255, 255))

            # Depth crosshair at box centre
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = 5
            draw.line([(cx - r, cy), (cx + r, cy)], fill=light, width=1)
            draw.line([(cx, cy - r), (cx, cy + r)], fill=light, width=1)

        return np.array(pil)
