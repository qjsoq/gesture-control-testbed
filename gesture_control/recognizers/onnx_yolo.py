from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel

from gesture_control.recognizers.gesture import GesturePrediction

logger = logging.getLogger(__name__)


class OnnxYoloRecognizerConfig(BaseModel):
    type: Literal["onnx_yolo"]
    model_path: Path
    confidence: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    providers: list[str] | None = None  # default = onnxruntime.get_available_providers()
    intra_op_num_threads: int = 4
    inter_op_num_threads: int = 1


class OnnxYoloRecognizer:
    """YOLOv8 ONNX recognizer using onnxruntime — no torch/ultralytics dependency."""

    def __init__(self, cfg: OnnxYoloRecognizerConfig) -> None:
        import onnxruntime as ort  # local import keeps deps optional

        if not cfg.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {cfg.model_path}")

        self._cfg = cfg

        so = ort.SessionOptions()
        so.intra_op_num_threads = cfg.intra_op_num_threads
        so.inter_op_num_threads = cfg.inter_op_num_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        available = ort.get_available_providers()
        # Prefer XNNPACK on ARM if present, then fall back to CPU
        preferred = cfg.providers or [
            p for p in ("XnnpackExecutionProvider", "CPUExecutionProvider") if p in available
        ]
        if not preferred:
            preferred = available

        self._session = ort.InferenceSession(
            str(cfg.model_path), sess_options=so, providers=preferred
        )
        self._input_name = self._session.get_inputs()[0].name
        self._input_shape = self._session.get_inputs()[0].shape  # [1, 3, H, W]
        self._names = self._load_names()

        logger.info(
            "onnx_yolo loaded model=%s classes=%d providers=%s input=%s threads=%d",
            cfg.model_path,
            len(self._names),
            self._session.get_providers(),
            self._input_shape,
            cfg.intra_op_num_threads,
        )

    @classmethod
    def from_config(cls, cfg: OnnxYoloRecognizerConfig) -> "OnnxYoloRecognizer":
        return cls(cfg)

    def _load_names(self) -> dict[int, str]:
        meta = self._session.get_modelmeta().custom_metadata_map
        raw = meta.get("names")
        if raw:
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    return {int(k): str(v) for k, v in parsed.items()}
            except (ValueError, SyntaxError):
                logger.warning("failed to parse 'names' metadata: %r", raw)
        logger.warning("ONNX metadata has no 'names'; falling back to numeric labels")
        return {}

    def predict(self, frame: np.ndarray) -> GesturePrediction | None:
        tensor, scale, pad = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: tensor})
        detections = self._postprocess(outputs[0], scale, pad, frame.shape[:2])
        if not detections:
            return None
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

    def _preprocess(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        imgsz = self._cfg.imgsz
        h, w = frame.shape[:2]
        scale = min(imgsz / h, imgsz / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        pad_x = (imgsz - nw) // 2
        pad_y = (imgsz - nh) // 2
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None]  # CHW + batch
        return tensor, scale, (pad_x, pad_y)

    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad: tuple[int, int],
        orig_shape: tuple[int, int],
    ) -> list[dict]:
        # Two known formats from ultralytics ONNX export:
        #   end2end (nms baked in):  [1, max_det, 6]  rows = [x1, y1, x2, y2, conf, class_id]
        #   raw YOLOv8:              [1, 4+nc, num_anchors]
        if output.ndim == 3 and output.shape[-1] == 6:
            return self._postprocess_end2end(output, scale, pad, orig_shape)
        return self._postprocess_raw(output, scale, pad, orig_shape)

    def _postprocess_end2end(
        self,
        output: np.ndarray,
        scale: float,
        pad: tuple[int, int],
        orig_shape: tuple[int, int],
    ) -> list[dict]:
        pad_x, pad_y = pad
        h_orig, w_orig = orig_shape
        rows = output[0]  # [max_det, 6]
        confs = rows[:, 4]
        keep = confs >= self._cfg.confidence
        if not keep.any():
            return []
        rows = rows[keep]

        detections: list[dict] = []
        for r in rows:
            x1, y1, x2, y2, conf, cls = r
            x1 = float(np.clip((x1 - pad_x) / scale, 0, w_orig - 1))
            y1 = float(np.clip((y1 - pad_y) / scale, 0, h_orig - 1))
            x2 = float(np.clip((x2 - pad_x) / scale, 0, w_orig - 1))
            y2 = float(np.clip((y2 - pad_y) / scale, 0, h_orig - 1))
            cid = int(cls)
            detections.append(
                {
                    "label": self._names.get(cid, str(cid)),
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2],
                }
            )
        return detections

    def _postprocess_raw(
        self,
        output: np.ndarray,
        scale: float,
        pad: tuple[int, int],
        orig_shape: tuple[int, int],
    ) -> list[dict]:
        preds = np.squeeze(output, axis=0).T
        if preds.shape[1] < 5:
            return []

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confs = class_scores[np.arange(len(class_scores)), class_ids]

        keep = confs >= self._cfg.confidence
        if not keep.any():
            return []
        boxes_xywh = boxes_xywh[keep]
        confs = confs[keep]
        class_ids = class_ids[keep]

        xy = boxes_xywh[:, :2]
        wh = boxes_xywh[:, 2:]
        x1y1 = xy - wh / 2
        x2y2 = xy + wh / 2
        boxes_xyxy = np.concatenate([x1y1, x2y2], axis=1)

        nms_boxes = np.concatenate([x1y1, wh], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            confs.tolist(),
            self._cfg.confidence,
            self._cfg.iou,
        )
        if len(indices) == 0:
            return []
        indices = np.array(indices).flatten()

        pad_x, pad_y = pad
        h_orig, w_orig = orig_shape

        detections: list[dict] = []
        for i in indices:
            x1, y1, x2, y2 = boxes_xyxy[i]
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            x1 = float(np.clip(x1, 0, w_orig - 1))
            y1 = float(np.clip(y1, 0, h_orig - 1))
            x2 = float(np.clip(x2, 0, w_orig - 1))
            y2 = float(np.clip(y2, 0, h_orig - 1))

            cid = int(class_ids[i])
            label = self._names.get(cid, str(cid))
            detections.append(
                {
                    "label": label,
                    "confidence": float(confs[i]),
                    "bbox": [x1, y1, x2, y2],
                }
            )
        return detections
