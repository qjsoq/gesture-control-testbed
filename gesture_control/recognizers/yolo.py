import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

import numpy as np
from gesture_control.recognizers.gesture import GesturePrediction

logger = logging.getLogger(__name__)


class YoloRecognizerConfig(BaseModel):
    type: Literal["yolo"]
    model_path: Path
    confidence: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: str = "cpu"  # "cpu", "cuda", "cuda:0", "0"


class YoloRecognizer:
    def __init__(self, cfg: YoloRecognizerConfig) -> None:
        from ultralytics import YOLO

        if not cfg.model_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {cfg.model_path}")
        self._cfg = cfg
        self._model = YOLO(str(cfg.model_path))
        self._names: dict[int, str] = self._model.names
        logger.info(
            "yolo loaded model=%s classes=%d device=%s",
            cfg.model_path,
            len(self._names),
            cfg.device,
        )

    @classmethod
    def from_config(cls, cfg: YoloRecognizerConfig) -> "YoloRecognizer":
        return cls(cfg)

    def predict(self, frame: np.ndarray) -> GesturePrediction | None:
        results = self._model.predict(
            frame,
            conf=self._cfg.confidence,
            iou=self._cfg.iou,
            imgsz=self._cfg.imgsz,
            device=self._cfg.device,
            verbose=False,
        )
        if not results:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        cls_arr = boxes.cls.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        xyxy_arr = boxes.xyxy.cpu().numpy()

        detections = [
            {
                "label": self._names[int(cls_arr[i])],
                "confidence": float(conf_arr[i]),
                "bbox": xyxy_arr[i].tolist(),
            }
            for i in range(len(boxes))
        ]
        top = max(detections, key=lambda d: d["confidence"])
        x1, y1, x2, y2 = top["bbox"]
        h, w = frame.shape[:2]
        cx = (x1 + x2) / 2.0 / w if w else None
        cy = (y1 + y2) / 2.0 / h if h else None
        return GesturePrediction(
            label=top["label"],
            confidence=top["confidence"],
            hand_x=cx,
            hand_y=cy,
            raw={"detections": detections},
        )
