import os
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key)

project = rf.workspace("vehicle-detection-ptdpb").project("traffic-vehicle-detection-fwht4")
version = project.version(7)

dataset = version.download("yolov8")

print("Dataset downloaded successfully.")