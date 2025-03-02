from proto import video_pb2


class FaceBox:
    """
      string label = 1;         // 类别标签
      //边界框坐标归一化比例（0.0~1.0）
      Point minPoint = 2;     // 左上角点
      Point maxPoint = 3;     // 右下角点
      float score = 4;          // 置信度
      string track_id = 5;      // 轨迹ID
      string face_id = 6;       // 人脸ID
      int64 expression_id = 7;  // 表情ID（0～6）
      string expression_feature = 8; // 表情特征
      string data = 9;          // 额外数据(JSON格式)
    """
    def __init__(self, label, minPoint, maxPoint, score, track_id, face_id, expression_id, expression_feature, data):
        self.label = label
        self.minPoint = minPoint
        self.maxPoint = maxPoint
        self.score = score
        self.track_id = track_id
        self.face_id = face_id
        self.expression_id = expression_id
        self.expression_feature = expression_feature
        self.data = data
        return video_pb2.FaceBox()