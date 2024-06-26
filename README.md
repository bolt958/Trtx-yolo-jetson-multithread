# Trtx-yolo-jetson-multithread<br>
多线程实现yolov5的trtx视频流推理<br>
线程1：视频文件输入or RTSP输入源<br>
线程2：预处理、推理、NMS<br>
线程3：渲染、保存、显示（其他应用...）<br>

usage : sudo   ./mtdet   xxx.engine   输入源


6-26   commit：<br>
1.新开线程解决无法实时显示检测视频的问题<br>
2.解决nano上无法保存检测视频的问题<br>
3.解决FPS计算错误问题并优化各种打印信息<br>
4.解决按键中断无法正常退出线程并释放资源问题<br>
5.修改了一些小bug<br>
