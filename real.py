# # import cv2
# # from ultralytics import YOLO

# # model = YOLO("C:/Users/DELL/Desktop/ultralytics/Datasets/Suspicious Activity Detection.v1i.yolov8/best_download.pt")

# # cap = cv2.VideoCapture(0)

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     results = model(frame, conf=0.3)
# #     annotated_frame = results[0].plot()

# #     cv2.imshow("Live Detection", annotated_frame)

# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()


# import cv2
# import logging
# from ultralytics import YOLO

# # Disable YOLO logs
# logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# model = YOLO("C:/Users/DELL/Desktop/ultralytics/Datasets/Suspicious Activity Detection.v1i.yolov8/best_download.pt")

# cap = cv2.VideoCapture(0)

# try:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model(frame, conf=0.3)
#         annotated_frame = results[0].plot()

#         cv2.imshow("YOLO Detection", annotated_frame)

#         # Press 'q' to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# except KeyboardInterrupt:
#     print("KeyboardInterrupt")

# finally:
#     cap.release()
#     cv2.destroyAllWindows()


import logging
import time

import cv2
import torch

from ultralytics import YOLO

# ----------------------------
# CONFIG
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# Suppress YOLO logs
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# Load your trained YOLO model
model = YOLO("C:/Users/DELL/Desktop/ultralytics/best_yolov8s.pt").to(device)

# ----------------------------
# CAMERA SETUP
# ----------------------------
cap = cv2.VideoCapture(0)  # 0 = default laptop camera
if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

print("📷 Camera started. Press 'q' to quit.")

# ----------------------------
# MAIN LOOP
# ----------------------------
try:
    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        # Resize for faster inference
        frame_resized = cv2.resize(frame, (640, 360))

        # Run YOLO inference
        results = model.predict(frame_resized, imgsz=320, device=device, verbose=False)

        # Draw YOLO annotations
        annotated = results[0].plot()

        # Calculate FPS
        fps = 1 / (time.time() - start_time)
        cv2.putText(annotated, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show in full screen
        cv2.namedWindow("YOLOv8 Detection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("YOLOv8 Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("YOLOv8 Detection", annotated)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n🛑 KeyboardInterrupt - Stopping.")

finally:
    cap.release()
    cv2.destroyAllWindows()
