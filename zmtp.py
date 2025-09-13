# Laptop (Receiver)
import zmq
import cv2
import numpy as np

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:5555")

while True:
    msg = socket.recv()
    nparr = np.frombuffer(msg, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is not None:
        cv2.imshow("Received", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break