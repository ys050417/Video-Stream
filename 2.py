import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
import sys

# -------------------------- 配置区 --------------------------
INPUT_RTSP = "rtsp://127.0.0.1:25544/input"
OUTPUT_RTSP = "rtsp://127.0.0.1:25544/output"
MODEL_WEIGHT = "yolo11n.pt"
YOLO_IMGSZ = 640
CONF_THRESH = 0.45
# 画质参数，越低画面越清晰
VIDEO_CRF = "16"
VIDEO_BITRATE = "3000k"
# -------------------------------------------------------------

# 加载YOLO11模型
model = YOLO(MODEL_WEIGHT)

# OpenCV读取流优化，降低缓存
cap = cv2.VideoCapture(INPUT_RTSP, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

if not cap.isOpened():
    print("错误：无法打开输入RTSP流！")
    exit(1)

# 获取视频原生分辨率、帧率
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"视频源信息：宽{w} × 高{h} | 帧率 {fps}fps")

# FFmpeg推流命令（高清编码）
ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-loglevel", "warning",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{w}x{h}",
    "-r", str(fps),
    "-i", "-",
    "-c:v", "libx264",
    "-preset", "medium",
    "-tune", "zerolatency",
    "-crf", VIDEO_CRF,
    "-b:v", VIDEO_BITRATE,
    "-maxrate", VIDEO_BITRATE,
    "-bufsize", "4000k",
    "-pix_fmt", "yuv420p",
    "-g", str(fps * 2),
    "-x264-params", "sync-lookahead=0:aq-mode=2",
    "-threads", "6",
    "-rtsp_transport", "tcp",
    "-rtsp_flags", "prefer_tcp",
    "-muxdelay", "0.01",
    "-fflags", "nobuffer",
    "-avioflags", "direct",
    "-f", "rtsp",
    OUTPUT_RTSP
]

# Windows系统隐藏FFmpeg黑窗口
start_info = None
if sys.platform == "win32":
    start_info = subprocess.STARTUPINFO()
    start_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW

# 启动FFmpeg推流进程
ff_proc = subprocess.Popen(
    ffmpeg_cmd,
    stdin=subprocess.PIPE,
    startupinfo=start_info
)
print(f"✅ 后台推流启动成功，输出地址：{OUTPUT_RTSP}")
print("程序无预览窗口，关闭PyCharm运行终端即可停止推流\n")

# 主循环处理帧（无本地预览）
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 读取视频流失败，程序退出")
        break

    # YOLO检测绘制
    results = model(frame, imgsz=YOLO_IMGSZ, conf=CONF_THRESH, verbose=False)
    draw_frame = results[0].plot(line_width=2, font_size=0.5)

    # 转灰度三通道送入FFmpeg
    gray_1ch = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray_1ch, cv2.COLOR_GRAY2BGR)

    # 管道推流
    try:
        ff_proc.stdin.write(gray_3ch.tobytes())
    except BrokenPipeError:
        print("❌ FFmpeg推流管道断开，结束运行")
        break

# 资源释放
cap.release()
if ff_proc.poll() is None:
    ff_proc.stdin.close()
    ff_proc.wait()
print("程序正常退出")