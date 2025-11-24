import cv2
from ultralytics import YOLO
import numpy as np

# --- SETTINGS ---
use_webcam = True  # Set to True for webcam, False for video file
video_path = r"C:\Users\Mohakk\Downloads\Best Fails of the Week _ Summer 2025 Fails.mp4"

# Load YOLOv8 model (make sure it's downloaded)
model = YOLO("yolov9t.pt")  # You can use yolov8s.pt / yolov8m.pt for better accuracy

# Open video capture source
cap = cv2.VideoCapture(0 if use_webcam else video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

window_name = "Human Detection - YOLOv8"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference and grab the first result
    results = model(frame)
    if len(results) == 0:
        annotated_frame = frame
    else:
        res = results[0]

        # Prepare a copy for drawing
        annotated_frame = frame.copy()
        person_count = 0

        # `res.boxes` contains detected boxes. Each box has xyxy, conf, cls
        try:
            boxes = res.boxes
            # convert to numpy arrays for iteration
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
            confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
            cls_ids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else []

            for (box, conf, cls_id) in zip(xyxy, confs, cls_ids):
                # try to resolve class name via model.names when available
                try:
                    name = model.names[int(cls_id)]
                except Exception:
                    # fallback: COCO person class is 0
                    name = "person" if int(cls_id) == 0 else str(cls_id)

                # Only draw persons
                if str(name).lower() == "person" or int(cls_id) == 0:
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{name} {conf:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    person_count += 1
        except Exception:
            # If boxes are in an unexpected format, fallback to the model plot
            annotated_frame = res.plot()
            person_count = 0

        # Show person count
        cv2.putText(annotated_frame, f"Persons: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow(window_name, annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
