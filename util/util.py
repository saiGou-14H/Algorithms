import cv2
import numpy as np
import psutil


def pixels_to_normalized(boxes, shape, mode="xyxy"):
    """
    将YOLOv8目标检测的边界框坐标转换为图像归一化比例（0.0~1.0）
    :param boxes: 边界框坐标，支持两种格式：
                  - "xyxy": [x_min, y_min, x_max, y_max] (绝对像素坐标)
                  - "xywh": [x_center, y_center, width, height] (归一化比例值)
    :param shape: 图像的 (width, height) 或 (height, width, channels)
    :param mode: 输入坐标模式，"xyxy" 或 "xywh" (默认"xyxy")
    :return: 归一化坐标，格式与输入模式一致，范围[0.0, 1.0]
    """
    # 解析图像尺寸
    if len(shape) == 3:  # OpenCV/PIL格式 (h, w, c)
        img_h, img_w = shape[0], shape[1]
    else:
        img_w, img_h = (shape[0], shape[1]) if len(shape) == 2 else (shape[0], shape[0])

    if mode == "xyxy":
        # 绝对像素坐标 -> 归一化比例
        x_min, y_min, x_max, y_max = boxes
        x_min_norm = x_min / img_w
        y_min_norm = y_min / img_h
        x_max_norm = x_max / img_w
        y_max_norm = y_max / img_h
        boxs = [x_min_norm, y_min_norm, x_max_norm, y_max_norm]
        return [round(x, 4) for x in boxs]
    elif mode == "xy":
        x_point , y_point = boxes
        x_point_norm = x_point / img_w
        y_point_norm = y_point / img_h
        boxs = [x_point_norm, y_point_norm]
        return [round(x, 4) for x in boxs]
    else:
        raise ValueError("Invalid mode. Use 'xyxy'.")


def normalized_to_pixels(boxes_norm, shape, mode="xyxy"):
    """
    将归一化边界框坐标还原为绝对像素坐标
    :param boxes_norm: 归一化坐标（范围 [0.0, 1.0]），格式为：
                       - "xyxy": [x_min_norm, y_min_norm, x_max_norm, y_max_norm]
                       - "xywh": [x_center_norm, y_center_norm, width_norm, height_norm]
    :param shape: 图像的 (width, height) 或 (height, width, channels)
    :param mode: 输入坐标模式，"xyxy" 或 "xywh" (默认"xyxy")
    :return: 绝对像素坐标，格式与输入模式一致
    """
    # 解析图像尺寸
    if len(shape) == 3:  # OpenCV/PIL格式 (h, w, c)
        img_h, img_w = shape[0], shape[1]
    else:
        img_w, img_h = (shape[0], shape[1]) if len(shape) == 2 else (shape[0], shape[0])

    # 边界保护函数
    def clip(value, max_val):
        return max(0, min(round(value), max_val))

    if mode == "xyxy":
        # 归一化坐标 -> 绝对像素坐标 (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = boxes_norm
        x_min_pixel = clip(x_min * img_w, img_w)
        y_min_pixel = clip(y_min * img_h, img_h)
        x_max_pixel = clip(x_max * img_w, img_w)
        y_max_pixel = clip(y_max * img_h, img_h)
        return [x_min_pixel, y_min_pixel, x_max_pixel, y_max_pixel]
    elif mode == "xy":
        x_center, y_center= boxes_norm
        x_center_pixel = clip(x_center * img_w, img_w)
        y_center_pixel = clip(y_center * img_h, img_h)
        return [x_center_pixel, y_center_pixel]
    else:
        raise ValueError("Invalid mode. Use 'xyxy' or 'xywh'.")
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


def get_cpu_usage(interval=1):
    cpu_usage = psutil.cpu_percent(interval=interval)
    print(f"当前CPU占用率：{cpu_usage}%")
    return cpu_usage
def put_points(pose_results, frame):
    keypoints = np.array(pose_results[0].keypoints.xy.int().tolist())
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