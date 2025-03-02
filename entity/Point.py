from proto import video_pb2


class Point(video_pb2.Point):
    """
    float x = 1;
    float y = 2;
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y