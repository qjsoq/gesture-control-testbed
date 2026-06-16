"""Export YOLO .pt to ONNX at a smaller imgsz for faster Pi inference.
Run on PC (where ultralytics is installed):
    python export_onnx.py
"""
from pathlib import Path

from ultralytics import YOLO

SRC = Path("D:/Downloads/best.pt")
IMGSZ = 320
OPSET = 12

if not SRC.exists():
    raise SystemExit(f"Source weights not found: {SRC}")

model = YOLO(str(SRC))
out = model.export(
    format="onnx",
    imgsz=IMGSZ,
    opset=OPSET,
    simplify=True,
    dynamic=False,
    half=False,
)
print(f"Exported: {out}")
