from proto import video_pb2


class PersonBox(video_pb2.PersonBox):
    """
      string label = 1;         // 类别标签
      Point minPoints = 2;     // 左上角点
      Point maxPoints = 3;     // 右下角点
      repeated Point points = 4; // 人体关键点
      float score = 5;          // 置信度
      string track_id = 6;      // 轨迹ID
      int64 attitude_id = 7;    // 姿态ID
      string attitude_feature = 8; // 姿态特征
      string data = 9;          // 额外数据(JSON格式)
    """
    def __init__(self, label, minPoint, maxPoint, points, score, track_id, attitude_id, attitude_feature, data):
        self.label = label
        self.minPoint = minPoint
        self.maxPoint = maxPoint
        self.points = points
        self.score = score
        self.track_id = track_id
        self.attitude_id = attitude_id
        self.attitude_feature = attitude_feature
        self.data = data