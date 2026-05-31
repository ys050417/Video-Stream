import cv2
import subprocess
import numpy as np

# ===================== 配置地址 =====================
# EasyDarwin 输入RTSP流（ffmpeg推流过来的地址）
rtsp_input_url = "rtsp://127.0.0.1:25544/input"
# 处理后输出RTSP流（推回EasyDarwin）
rtsp_output_url = "rtsp://127.0.0.1:25544/output"

# ===================== FFmpeg推流命令 =====================
ffmpeg_cmd = [
    'ffmpeg',
    '-y',  # 覆盖已存在文件
    '-f', 'rawvideo',
    '-pix_fmt', 'gray',  # 输入为灰度格式
    '-s', '1280x720',    # 视频分辨率（和原视频一致）
    '-i', '-',           # 从标准输入读取帧数据
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-crf', '28',        # 画质参数
    '-threads', '4',
    '-f', 'rtsp',
    rtsp_output_url
]

# ===================== 打开视频流 =====================
cap = cv2.VideoCapture(rtsp_input_url)
if not cap.isOpened():
    print(f"无法打开视频流: {rtsp_input_url}")
    exit()

# ===================== 启动FFmpeg进程 =====================
process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# ===================== 实时处理循环 =====================
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧，退出")
        break

    # 核心：彩色帧转灰度
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 把灰度帧写入FFmpeg推流
    try:
        process.stdin.write(gray_frame.tobytes())
    except BrokenPipeError:
        print("FFmpeg进程已关闭")
        break

    # 本地预览（可选）
    # cv2.imshow("Processed Frame", gray_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# ===================== 资源释放 =====================
cap.release()
cv2.destroyAllWindows()
if process.poll() is None:
    process.stdin.close()
    process.wait()