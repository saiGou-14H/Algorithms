import psutil
import time


def get_cpu_usage(interval=1):
    while True:
        # 获取CPU的使用率
        cpu_usage = psutil.cpu_percent(interval=interval)
        print(f"当前CPU占用率：{cpu_usage}%")

        # 每隔1秒输出一次
        time.sleep(interval)


if __name__ == "__main__":
    get_cpu_usage()