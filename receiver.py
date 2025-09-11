# detection_receiver.py

import logging

import cv2
import numpy as np
import torch
import zmq

from ultralytics import YOLO

# ----------------------------
# CONFIG
# ----------------------------
ZMQ_ADDRESS = "tcp://*:5555"  # Receiver binds here
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# Suppress YOLO logs
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# Load your trained YOLO model
model = YOLO("C:/Users/DELL/Desktop/ultralytics/Datasets/Suspicious Activity Detection.v1i.yolov8/best_download.pt").to(
    device
)

# ----------------------------
# SETUP ZMQ
# ----------------------------
context = zmq.Context()
socket = context.socket(zmq.PULL)  # <-- PULL to match Pi's PUSH
socket.bind(ZMQ_ADDRESS)

print("📡 Receiver started, waiting for frames...")

# ----------------------------
# MAIN LOOP
# ----------------------------
try:
    while True:
        msg = socket.recv()  # Blocking receive

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
            imgsz=320,  # lower input size = faster inference
            device=device,
            verbose=False,
        )

        # Draw YOLO annotations
        annotated = results[0].plot()

        # Show in full screen
        cv2.namedWindow("YOLOv8 Detection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("YOLOv8 Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("YOLOv8 Detection", annotated)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n🛑 KeyboardInterrupt - Receiver stopped.")

finally:
    cv2.destroyAllWindows()
    socket.close()
    context.term()
