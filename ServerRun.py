import random
import threading
import queue
import time

import cv2
import numpy as np
from concurrent import futures
import grpc
import psutil
import torch

import service
from proto import video_pb2, video_pb2_grpc

# 根据CPU核心数动态调整
import os

from service.detect import detect_face, detect_pose
from ultralytics import YOLO
from util.util import put_points, normalized_to_pixels, pixels_to_normalized

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
        self.executor = futures.ThreadPoolExecutor(max_workers=cpu_cores*2)  # 根据CPU核心数动态调整


    def ProcessFrame(self, request_iterator, context):
        client_id = context.peer()
        print(f"客户端 {client_id} 连接建立")
        # 为每个客户端创建独立的结果队列
        with self.lock:
            self.result_queues[client_id] = queue.Queue(maxsize=5*60)

        # 启动异步处理线程
        processing_thread = threading.Thread(
            target=self._async_process,
            args=(request_iterator, client_id),
            daemon=True
        )
        processing_thread.start()

        # 流式发送结果
        try:
            while context.is_active():
                try:
                    result = self.result_queues[client_id].get(timeout=0.1)
                    yield result
                except queue.Empty:
                    continue
        finally:
            self._cleanup_client(client_id)

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
            img = np.frombuffer(frame.image_data, dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            if img is None:
                print("解码失败")
                return None
            face_boxes = detect_face(img)
            person_boxes = detect_pose(img)
            end_time = time.time()

            img = cv2.imencode('.jpg', img)[1].tobytes()
            print(f"算法耗时: {end_time - start_time}")
            return video_pb2.AnalysisResult(
                timestamp=frame.timestamp,
                image_data=img,
                face_boxes=face_boxes,
                person_boxes=person_boxes,
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

    cpu_cores = os.cpu_count()
    print(f"CPU核心数: {cpu_cores}")  # 16
    serve()