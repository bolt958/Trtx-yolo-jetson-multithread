# Trtx-yolo-jetson-multithread<br>
多线程实现yolov5的trtx视频流推理<br>
线程1：视频文件输入or RTSP输入源<br>
线程2：预处理、推理、NMS<br>
线程3：渲染、保存、（其他应用...）<br>
线程4：显示<br>

usage : sudo   ./mtdet   xxx.engine   输入源
<br>

6-26 ：<br>
1.解决新开线程解决无法实时显示检测视频的问题<br>
2.解决nano上无法保存检测视频的问题<br>
3.解决FPS计算错误问题并优化各种打印信息<br>
4.解决按键中断无法正常退出线程并释放资源问题<br>
5.修改了一些小bug<br>
<br>
6-27:<br>
1.读取视频文件中断异常无法正常退出<br>
2.自启动不能加入显示线程，否则会dumped<br>

6-28:<br>
暂时注释掉了FPS计算：原来的1000/process_all_time不对，因为多线程并发处理并非是每个步骤消耗时间的总和，这样算出来的FPS比实际的小<br>
exp:nano上FPS显示10左右，但实际上每分钟能处理近900帧（即FPS≈15）
