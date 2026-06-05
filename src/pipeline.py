import cv2
import math
from collections import defaultdict, deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import config


def process_video(input_path, output_path):

    model = YOLO(config.MODEL_PATH)
    tracker = DeepSort(max_age=30, n_init=3)

    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25
    fps = int(fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    track_history = defaultdict(lambda: deque(maxlen=10))
    vehicle_count = defaultdict(int)
    counted_ids = set()
    overspeed_count = 0
    violated_ids = set()

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if conf >= config.CONF_THRES:
                w = x2 - x1
                h = y2 - y1

                name = model.names[cls_id]
                name = config.CLASS_REMAP.get(name, name)

                detections.append(([x1, y1, w, h], conf, name))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            obj_class = track.get_det_class()

            cx = (x1 + x2)//2
            cy = (y1 + y2)//2

            track_history[track_id].append((cx, cy))

            if len(track_history[track_id]) >= 2:
                p1 = track_history[track_id][-2]
                p2 = track_history[track_id][-1]

                pixel_dist = math.dist(p1, p2)
                speed = pixel_dist * config.PIXEL_TO_METER * fps * 3.6
            else:
                speed = 0

            color = (0,0,255) if speed > config.SPEED_LIMIT else (0,255,0)
            label = f"{obj_class} ID:{track_id} {int(speed)} km/h"

            if speed > config.SPEED_LIMIT:
                 color = (0, 0, 255)
                 violation_text = "OVERSPEED"
                 if track_id not in violated_ids:
                      overspeed_count += 1
                      violated_ids.add(track_id)
            else:
                 color = (0, 255, 0)
                 violation_text = ""

            label = f"{obj_class} ID:{track_id} {int(speed)} km/h {violation_text}"

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

            if cy > config.COUNT_LINE_Y and track_id not in counted_ids:
                vehicle_count[obj_class] += 1
                counted_ids.add(track_id)

        cv2.line(frame,(0,config.COUNT_LINE_Y),
                 (width,config.COUNT_LINE_Y),(255,0,0),2)

        frame = cv2.resize(frame, (width, height))
        out.write(frame)

    cap.release()
    out.release()

    return output_path, dict(vehicle_count), overspeed_count