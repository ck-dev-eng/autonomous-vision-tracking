import cv2
import numpy as np
from ultralytics import YOLO

class PerceptionEngine:
    def __init__(self):
        # YOLOv8 nano modelini yükle
        self.model = YOLO('yolov8n.pt') 
        
    def process_frame(self, frame, conf_threshold=0.35):
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        alert_triggered = False
        detected_count = 0
        processed_img = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                target_classes = [0, 1, 2, 3, 5, 7]
                if cls in target_classes:
                    detected_count += 1
                    
                    box_area = (x2 - x1) * (y2 - y1)
                    frame_area = frame.shape[0] * frame.shape[1]
                    
                    if box_area / frame_area > 0.15:
                        alert_triggered = True
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)
                        
                    cv2.rectangle(processed_img, (x1, y1), (x2, y2), color, 2)
                    label = f"{self.model.names[cls]} {conf:.2f}"
                    cv2.putText(processed_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
        return processed_img, alert_triggered, detected_count
