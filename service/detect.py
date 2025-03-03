import random

import cv2

from proto import video_pb2
from service import face, device, expression, pose
from util.util import normalized_to_pixels, put_points


def detect_expression(imGray):
    result = expression.predict(source=imGray,device=device,verbose=False)
    names_dict = result[0].names
    expression_id = result[0].probs.top1
    expression_feature = names_dict[expression_id]
    return expression_id,expression_feature

def detect_face(img):
    face_boxes = []
    results = face.predict(img, device=device, imgsz = 1984,conf=0.4, verbose=False)
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxyn.tolist()[0]
        minPoint = video_pb2.Point(x=x1, y=y1)
        maxPoint = video_pb2.Point(x=x2, y=y2)
        x3, y3, x4, y4 = normalized_to_pixels([x1, y1, x2, y2], img.shape)
        imGray = cv2.cvtColor(img[y3:y4, x3:x4], cv2.COLOR_BGR2GRAY)
        expression_id,expression_feature = detect_expression(imGray)
        cv2.rectangle(img, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.putText(img, f'{expression_feature}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (36, 255, 12), 2)
        face_boxe = video_pb2.FaceBox(
            label='face',
            score=box.conf.item(),
            track_id=random.randint(0, 100),
            face_id=random.randint(0, 100),
            expression_id=expression_id,
            expression_feature=expression_feature,
            minPoint=minPoint,
            maxPoint=maxPoint,
        )
        face_boxes.append(face_boxe)
    return face_boxes

def detect_pose(img):
    person_boxes = []
    pose_results = pose.predict(img, device=device, imgsz = 1984,conf=0.4, verbose=False)
    if pose_results[0].keypoints.conf != None:
        keypoints = pose_results[0].keypoints.xyn.tolist()
        boxes = pose_results[0].boxes.xyxyn.tolist()
        for box, keypoint in zip(boxes, keypoints):
            x1, y1, x2, y2 = box
            minPoint = video_pb2.Point(x=x1, y=y1)
            maxPoint = video_pb2.Point(x=x2, y=y2)
            points = []
            for point in keypoint:
                points.append(video_pb2.Point(x=point[0], y=point[1]))
            person = video_pb2.PersonBox(
                label='person',
                score=1,
                track_id=random.randint(0, 100),
                attitude_id=random.randint(0, 100),
                attitude_feature='姿态',
                minPoint=minPoint,
                maxPoint=maxPoint,
                points=points,
            )
            person_boxes.append(person)
        put_points(pose_results, img)
        return person_boxes