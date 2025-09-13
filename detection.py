# detection_receiver.py

import cv2
import zmq
import numpy as np
import torch
import logging
from ultralytics import YOLO

# ----------------------------
# CONFIG
# ----------------------------
ZMQ_ADDRESS = "tcp://*:5555"   # Receiver binds here
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# Suppress YOLO logs
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# Load your trained YOLO model (use raw string for Windows paths)
model = YOLO(r"C:\Users\DELL\Desktop\ultralytics\best_yolov8s.pt").to(device)

# === Configure which classes are considered "suspicious" ===
# You can fill either of these. If both empty, script will auto-match suspicious keywords in class names.
SUSPICIOUS_CLASS_IDS = []         # e.g. [0, 2]  (class indices)
SUSPICIOUS_CLASS_NAMES = []       # e.g. ["intruder", "fight", "suspicious"]

# ----------------------------
# SETUP ZMQ
# ----------------------------
context = zmq.Context()
socket = context.socket(zmq.PULL)   # <-- PULL to match Pi's PUSH
socket.bind(ZMQ_ADDRESS)

print("📡 Receiver started, waiting for frames...")

alert_count = 0

# ----------------------------
# MAIN LOOP
# ----------------------------
try:
    while True:
        msg = socket.recv()   # Blocking receive

        # Decode JPEG bytes
        nparr = np.frombuffer(msg, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # Resize for faster inference
        frame_resized = cv2.resize(frame, (640, 360))

        # Run YOLOv8 inference
        results = model.predict(
            frame_resized,
            imgsz=320,       # lower input size = faster inference
            device=device,
            verbose=False
        )

        # Draw YOLO annotations (annotated is BGR numpy image)
        annotated = results[0].plot()

        # Determine detected classes (IDs and names)
        detected_ids = []
        detected_names = []
        try:
            num_boxes = len(results[0].boxes)
        except Exception:
            num_boxes = 0

        if num_boxes > 0:
            cls_tensor = results[0].boxes.cls  # may be tensor
            try:
                cls_array = cls_tensor.cpu().numpy().astype(int)
            except Exception:
                # fallback if already numpy or list-like
                cls_array = np.array([int(x) for x in cls_tensor])
            for cid in cls_array:
                detected_ids.append(int(cid))
                # get name safely from model.names
                name = str(model.names[int(cid)]) if int(cid) in model.names else str(int(cid))
                detected_names.append(name)

        # Decide whether a suspicious detection occurred
        suspicious_detected = False

        # check by id first if provided
        if SUSPICIOUS_CLASS_IDS and any(cid in SUSPICIOUS_CLASS_IDS for cid in detected_ids):
            suspicious_detected = True

        # check by name if provided
        if not suspicious_detected and SUSPICIOUS_CLASS_NAMES:
            for dname in detected_names:
                if any(sname.lower() in dname.lower() for sname in SUSPICIOUS_CLASS_NAMES):
                    suspicious_detected = True
                    break

        # If user didn't configure suspicious classes, try a keyword match on detected class names
        if not suspicious_detected and not SUSPICIOUS_CLASS_IDS and not SUSPICIOUS_CLASS_NAMES:
            for dname in detected_names:
                if any(k in dname.lower() for k in ("suspicious", "suspect", "intruder", "fight", "violence")):
                    suspicious_detected = True
                    break

        # If suspicious -> overlay alert message on the annotated frame
        if suspicious_detected:
            alert_count += 1
            alert_text = "🚨 ALERT: Suspicious Activity Detected!"
            # compute text size
            (tw, th), baseline = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            padding = 12
            # draw filled red rectangle as background for text (top-left)
            cv2.rectangle(annotated, (10, 10), (10 + tw + padding, 10 + th + padding), (0, 0, 255), -1)
            # put white text over it
            cv2.putText(annotated, alert_text, (16, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # show alert count on top-right
            right_text = f"Alerts: {alert_count}"
            (rw, rh), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (w - rw - 26, 10), (w - 10, 10 + rh + 8), (0, 0, 128), -1)
            cv2.putText(annotated, right_text, (w - rw - 18, 10 + rh), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Show in full screen
        cv2.namedWindow("YOLOv8 Detection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("YOLOv8 Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("YOLOv8 Detection", annotated)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 KeyboardInterrupt - Receiver stopped.")

finally:
    cv2.destroyAllWindows()
    socket.close()
    context.term()
