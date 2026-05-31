import cv2
import subprocess
import numpy as np
import base64
import time
import sys
import threading
from ultralytics import YOLO
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ======================== 配置区域 ========================
RTSP_INPUT  = "rtsp://127.0.0.1:25544/input"
RTSP_OUTPUT = "rtsp://127.0.0.1:25544/output"
YOLO_WEIGHT = "yolo11n.pt"

# 阿里云百炼
API_KEY    = "sk-e54090d370ef4979901cf2ea6f434ffe"
BASE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.6-plus"

AI_INTERVAL = 10
# ==========================================================

# 全局帧（线程安全）
latest_frame = None
frame_lock = threading.Lock()

# 加载模型
yolo = YOLO(YOLO_WEIGHT)
llm = ChatOpenAI(
    openai_api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    temperature=0.1,
    max_tokens=128,
    timeout=45
)

# 帧转base64（缩小图，防止超时）
def frame_to_base64(frame):
    small_frame = cv2.resize(frame, (640, 360))
    _, buf = cv2.imencode(".jpg", small_frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
    b64 = base64.b64encode(buf).decode()
    return f"data:image/jpeg;base64,{b64}"

# ---------------------- AI 后台线程 ----------------------
def ai_analyze_thread():
    global latest_frame
    while True:
        time.sleep(AI_INTERVAL)
        try:
            with frame_lock:
                if latest_frame is None:
                    continue
                frame = latest_frame.copy()

            img_url = frame_to_base64(frame)
            msg = HumanMessage(content=[
                {"type": "text", "text": "简洁描述画面内容，80字以内"},
                {"type": "image_url", "image_url": {"url": img_url}}
            ])
            res = llm.invoke([msg])
            print("\n【AI分析】", res.content)
        except Exception as e:
            print("\n【AI】调用失败（网络/超时）:", str(e)[:60])

# ---------------------- 主线程：YOLO + 超清推流 ----------------------
def main_stream():
    global latest_frame
    cap = cv2.VideoCapture(RTSP_INPUT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # FFmpeg 超清 + 流畅 + 不卡顿 + 无马赛克
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", "30",
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-crf", "14",
        "-b:v", "4500k",
        "-maxrate", "4500k",
        "-bufsize", "6000k",
        "-pix_fmt", "yuv420p",
        "-rtsp_transport", "tcp",
        "-f", "rtsp", RTSP_OUTPUT
    ]

    # 隐藏黑窗口
    startupinfo = None
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    ff = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, startupinfo=startupinfo)

    print("=============================================")
    print("✅ 启动成功：YOLO11n检测 + RTSP高清推流 + AI后台分析")
    print("=============================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 给AI提供最新帧
        with frame_lock:
            latest_frame = frame

        # YOLO 检测
        results = yolo(frame, imgsz=480, conf=0.4, verbose=False)
        draw = results[0].plot(line_width=2, font_size=0.5)

        # 灰度处理
        gray = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 推流
        try:
            ff.stdin.write(gray3.tobytes())
        except:
            break

    cap.release()
    ff.stdin.close()
    ff.wait()

# ---------------------- 启动 ----------------------
if __name__ == "__main__":
    threading.Thread(target=ai_analyze_thread, daemon=True).start()
    main_stream()