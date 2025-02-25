import random
import threading
import queue
import time

import cv2
import numpy as np
from concurrent import futures
import grpc
import psutil

from entity import video_pb2, video_pb2_grpc

# 根据CPU核心数动态调整
import os

from ultralytics import YOLO
from util.util import put_points

face = YOLO("service/yolov8n-pose-face.pt")
bq = YOLO("service/bq-best.pt")
pose = YOLO("service/best.pt")

face.predict("loss.png")
bq.predict("loss.png")
pose.predict("loss.png")
cpu_cores = os.cpu_count()
print(f"CPU核心数: {cpu_cores}")#16
def get_cpu_usage(interval=1):
    cpu_usage = psutil.cpu_percent(interval=interval)
    print(f"当前CPU占用率：{cpu_usage}%")
    return cpu_usage
class AsyncVideoProcessor(video_pb2_grpc.VideoProcessorServicer):
    def __init__(self):
        # 初始化线程安全的队列
        self.task_queue = queue.Queue(maxsize=300)  # 限制处理队列长度
        self.result_queues = {}  # 为每个客户端维护独立的结果队列
        self.lock = threading.Lock()

        # 启动处理线程池
        self.executor = futures.ThreadPoolExecutor(max_workers=4)  # 根据CPU核心数动态调整


    def ProcessFrame(self, request_iterator, context):
        client_id = context.peer()
        print(f"客户端 {client_id} 连接建立")
        for request in request_iterator:
            return self._process_single_frame(request, client_id)

    def _async_process(self, request_iterator, client_id):
        """异步处理流水线"""
        try:
            for frame in request_iterator:
                # 提交任务到线程池
                future = self.executor.submit(
                    self._process_single_frame,
                    frame,
                    client_id
                )
                # 非阻塞添加结果回调
                future.add_done_callback(
                    lambda f: self._handle_result(f, client_id)
                )
        except Exception as e:
            print(f"处理异常: {str(e)}")
        finally:
            print(f"客户端 {client_id} 处理结束")

    def _process_single_frame(self, frame, client_id):

        start_time = time.time()
        """单帧处理逻辑"""
        try:
            # print(np.frombuffer(frame.image_data, dtype=np.uint8).shape)
            # 硬件加速解码JPEG
            img = np.frombuffer(frame.image_data, dtype=np.uint8)

            if img is None:
                print("解码失败")
                return None
            #计算解码耗时
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            #加水印
            results = face.predict(img, device='cpu', conf=0.4)
            pose_results = pose.predict(img, device='cpu', conf=0.4)
            boxes = results[0].boxes.xyxy.int().tolist()
            for box in boxes:
                x1, y1, x2, y2 = box
                imGray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                bq_results = bq.predict(source=imGray, device='cpu', show_labels=True)
                names_dict = bq_results[0].names
                probs = bq_results[0].probs.data.tolist()
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{names_dict[np.argmax(probs)]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (36, 255, 12), 2)

            if pose_results[0].keypoints.conf != None:
                put_points(pose_results, img)

            end_time = time.time()
            # 模拟算法随机耗时
            # time.sleep(random.randint(1, 2))
            # time.sleep(0.5)
            # 转回bytestring

            img = cv2.imencode('.jpg', img)[1].tobytes()

            print(f"算法耗时: {end_time - start_time}")


            return video_pb2.AnalysisResult(
                timestamp=frame.timestamp,
                image_data=img,
                boxes=[video_pb2.BoundingBox(
                    label='person',
                    x=100, y=100,
                    width=50, height=50,
                    score=0.9
                )]
            )
        except Exception as e:
            print(f"帧处理异常: {str(e)}")
            return None

    def _handle_result(self, future, client_id):
        """处理完成回调"""
        try:
            result = future.result()
            if result is not None:
                self.result_queues[client_id].put(result)
        except Exception as e:
            print(f"结果处理异常: {str(e)}")

    def _cleanup_client(self, client_id):
        """客户端连接清理"""
        with self.lock:
            cv2.destroyWindow(f'Client-{client_id}')
            del self.result_queues[client_id]
            print(f"客户端 {client_id} 资源已释放")


def serve():
    # 配置高性能服务器参数
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=cpu_cores * 4),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
            ('grpc.so_reuseport', 1),
            ('grpc.http2.max_pings_without_data', 0),
        ]
    )

    video_pb2_grpc.add_VideoProcessorServicer_to_server(
        AsyncVideoProcessor(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("服务端已启动（异步流模式），监听50051端口...")
    server.wait_for_termination()


if __name__ == '__main__':
    cv2.destroyAllWindows()
    serve()