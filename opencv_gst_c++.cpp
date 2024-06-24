#include <opencv2/opencv.hpp>
#include <iostream>


int main()
{
    std::string rtsp_url = "rtsp:/"; 
    std::string pipeline = "rtspsrc location=" + rtsp_url + " ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=960, height=540, format=BGRx ! videoconvert ! appsink sync=false";
    std::cout << pipeline << std::endl;

    cv::VideoCapture cap = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);


    if(!cap.isOpened())
    {
        std::cerr << "error to open camera." << std::endl;
        return -1;
    }
    std::cout << cv::getBuildInformation() << std::endl;
    cv::Mat frame ;


    while(1)
    {
        bool ret = cap.read(frame);
        if(ret)
        {
            cv::imshow("Frame",frame);
            if (cv::waitKey(1) == 27)   
                break;
        }

    }
    cv::destroyWindow("Frame");
    cap.release();
    return 0;
}
