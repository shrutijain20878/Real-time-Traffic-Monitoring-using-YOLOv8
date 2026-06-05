from ultralytics import YOLO
import torch

model = YOLO("yolov8n.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"

model.train(
    data="traffic-vehicle-detection-7/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=device
)

metrics = model.val(
    data="traffic-vehicle-detection-7/data.yaml"
)

print(metrics)