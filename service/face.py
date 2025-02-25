import cv2
from ultralytics import YOLO

def detect_face(img):
    results = face.predict(img, device='cpu', conf=0.4)
    boxes = results[0].boxes.xyxy.int().tolist()
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img
