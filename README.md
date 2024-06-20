# Trtx-yolo-jetson-multithread<br>
多线程实现yolov5的trtx视频流推理<br>
线程1：视频文件输入or RTSP输入源<br>
线程2：预处理、推理、NMS<br>
线程3：渲染、保存、显示（其他应用...）<br>
