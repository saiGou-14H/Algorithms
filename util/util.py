import cv2
import numpy as np


def line(pre_idx, idx, points, frame, x=0):
    rule = [[2 + x, 4 + x], [2 + x, 0], [6 + x, 8 + x, 10 + x], [6 + x, 12 + x, 14 + x, 16 + x], [4 + x, 6 + x],
            [6 + x, 10 + x], [12 + x, 16 + x]]
    point_size = 3
    point_color = (225, 249, 154)  # BGR
    thickness = 4  # 可以为 0 、4、8
    nex_point = points[idx]
    if nex_point[0] != 0 and nex_point[1] != 0:
        cv2.circle(frame, points[idx], point_size, point_color, thickness)
        if pre_idx != None and points[pre_idx][0] != 0 and points[pre_idx][1] != 0:
            cv2.line(frame, points[pre_idx], nex_point, point_color, 1)

    for idxs in rule:
        for _id in idxs:
            if _id == idx:
                if len(idxs) > idxs.index(_id) + 1:
                    line(idx, idxs[idxs.index(_id) + 1], points, frame, x)


def put_points(pose_results, frame):
    keypoints = np.array(pose_results[0].keypoints.xy.int().cpu().tolist())
    point_color = (225, 249, 154)  # BGR
    for keypoint in keypoints:
        line(None, 1, keypoint, frame, -1)
        line(None, 5, keypoint, frame, -1)
        line(None, 2, keypoint, frame)
        line(None, 6, keypoint, frame)
        if keypoint[5][0] != 0 and keypoint[5][1] != 0 and keypoint[6][0] != 0 and keypoint[6][1] != 0:
            cv2.line(frame, keypoint[5], keypoint[6], point_color, 1)
        if keypoint[11][0] != 0 and keypoint[11][1] != 0 and keypoint[12][0] != 0 and keypoint[12][1] != 0:
            cv2.line(frame, keypoint[11], keypoint[12], point_color, 1)