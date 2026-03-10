import torch

VIDEO_PATH = "data/traffic.mp4"
OUTPUT_VIDEO = "output/final_traffic_monitoring.avi"

MODEL_PATH = "models/best.pt"

CONF_THRES = 0.25

PIXEL_TO_METER = 0.05

# Speed limit for violation
SPEED_LIMIT = 60

# Counting line position
COUNT_LINE_Y = 400

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_REMAP = {
    "bikes": "bike",
    "bus": "bus",
    "car": "car",
    "truck": "truck"
}