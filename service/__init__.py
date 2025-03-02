import torch

from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face = YOLO("service/yolov8n-pose-face.pt").to(device)
expression = YOLO("service/bq-best.pt").to(device)
pose = YOLO("service/best.pt").to(device)