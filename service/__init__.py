import torch

from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# face = YOLO("service/yolov8-pose-p6-HGStem-LSCD-BIFPN-Rep1.yaml").load("service/face.pt").to(device)
face = YOLO("service/face.pt").to(device)
expression = YOLO("service/bq-best.pt").to(device)
pose = YOLO("service/yolov8n-pose.pt").to(device)