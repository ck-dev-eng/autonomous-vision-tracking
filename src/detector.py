import cv2
import numpy as np
from ultralytics import YOLO

class AutonomousDetector:
    def __init__(self, model_version='yolov8n.pt'):
        self.model = YOLO(model_version)
        self.target_classes = [0, 1, 2, 3, 5, 7]

    def process_frame(self, frame, conf_threshold=0.4):
        results = self.model(frame, conf=conf_threshold, verbose=False)[0]
        detected_objects = {"car": 0, "person": 0, "truck/bus": 0, "two-wheeler": 0}
        collision_warning = False
        frame_h, frame_w, _ = frame.shape

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.target_classes:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            label = self.model.names[cls_id]
            if cls_id in [2]:
                detected_objects["car"] += 1
            elif cls_id in [0]:
                detected_objects["person"] += 1
            elif cls_id in [5, 7]:
                detected_objects["truck/bus"] += 1
            elif cls_id in [1, 3]:
                detected_objects["two-wheeler"] += 1

            box_area = (x2 - x1) * (y2 - y1)
            frame_area = frame_h * frame_w
            area_ratio = box_area / frame_area

            if area_ratio > 0.25 and (x1 < frame_w * 0.7 and x2 > frame_w * 0.3):
                collision_warning = True
                box_color = (0, 0, 255)
            else:
                box_color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            caption = f"{label.capitalize()} {conf:.2f}"
            cv2.putText(frame, caption, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        if collision_warning:
            cv2.putText(frame, "WARNING: COLLISION RISK AHEAD!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        return frame, detected_objects, collision_warning
