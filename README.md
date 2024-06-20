# Trtx-yolo-jetson-multithread
多线程实现yolov5的trtx视频流推理
线程1：视频文件输入or RTSP输入源
线程2：预处理、推理、NMS
线程3：渲染、保存、显示（其他应用...）
