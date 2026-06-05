# test_pipeline.py
from src.pipeline import process_video

output, counts = process_video("traffic.mp4", "output.mp4")

print("Done:", output)
print("Counts:", counts)