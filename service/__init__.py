import torch

from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face = YOLO("service/face.pt").to(device)
expression = YOLO("service/bq-pose_best.pt").to(device)
pose = YOLO("service/yolov8n-pose.pt").to(device)