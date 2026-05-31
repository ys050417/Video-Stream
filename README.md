# RTSP 视频流实时智能处理

该项目基于 OpenCV + FFmpeg + EasyDarwin + YOLO11 + 通义千问视觉大模型 构建的低延迟、实时、可部署视频流处理平台，支持 RTSP 拉流、实时目标检测、AI 视觉分析、灰度渲染、高清 RTSP 推流全链路功能。

## 一、核心能力

- **RTSP 流低延迟拉取**

-  **实时灰度处理**

-  **YOLO11 目标检测**

- **多线程视觉大模型分析**

- **FFmpeg 硬编码推流**

- **后台静默运行**

-  **7×24 小时稳定推流**

## 二、环境部署

### （一）核心图像处理

```
pip install opencv-python==4.8.0.74
pip install numpy==1.24.3
```

### （二）YOLO11目标检测

```
pip install ultralytics==8.2.0
```

### （三）大模型调用

```
pip install langchain_openai==0.1.0
pip install openai==1.13.0
```

## 三、运行

### （一）启动 EasyDarwin

- 运行 `EasyDarwin.exe`，保持后台开启

### （二）向 input 地址推送测试视频

```
ffmpeg -re -stream_loop -1 -i test.mp4 -c copy -f rtsp rtsp://127.0.0.1:25544/input
```

- `-re`：按原始帧率推送

- `-stream_loop -1`：无限循环

- `test.mp4`：替换为你的本地视频

### （三）运行处理脚本

- 基础灰度推流

```
python 1.py
```

-  YOLO 检测 + 高清推流

```
python 2.py
```

-  YOLO + 视觉大模型分析

```
 python 3.py
```

### （四）查看输出流

- VLC → 媒体 → 打开网络串流 → 输入

```
rtsp://127.0.0.1:25544/output
```

## 四、API调用

```
API_KEY    = "sk-xxxx"   # 阿里云百炼 API Key
BASE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.6-plus"
AI_INTERVAL = 10         # 每10秒分析一次
```