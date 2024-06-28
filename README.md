# Trtx-yolo-jetson-multithread<br>
多线程实现yolov5的trtx视频流推理<br>
线程1：视频文件输入or RTSP输入源<br>
线程2：预处理、推理、NMS<br>
线程3：渲染、保存、（其他应用...）<br>
线程4：显示<br>

usage : sudo   ./mtdet   xxx.engine   输入源
<br>

6-26 ：<br>
1.新开线程解决无法实时显示检测视频的问题<br>
2.解决nano上无法保存检测视频的问题<br>
3.解决FPS计算错误问题并优化各种打印信息<br>
4.解决按键中断无法正常退出线程并释放资源问题<br>
5.修改了一些小bug<br>
<br>
6-27:<br>
1.读取视频文件中断异常无法正常退出<br>
2.自启动不能加入显示线程，否则会dumped<br>

6-28:<br>
暂时注释掉了FPS计算：原来的1000/process_all_time不对，因为多线程并发处理并非是每个步骤消耗时间的总和<br>
这样算出来的FPS,在输入帧率<推理帧率的设备上，比实际的小，在输入帧率>推理帧率的设备上，比实际的大<br>
exp:nano上FPS显示10左右，但实际上每分钟能处理近900帧（即FPS≈15） 4070s上FPS显示300-500左右，但实际11秒钟仅处理1000帧（即FPS约等于100）<br>
注：两个设备上任务不同

6-28:<br>
更新了FPS计算方式：<br>
计算累计30帧的总处理时间，然后除以30得到平均每帧的处理时间，再用1000除以这个时间得到FPS，即每30帧更新一次FPS
