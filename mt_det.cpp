#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include "preprocess.h"
#include <iomanip>
#include <sstream>
#include <fstream>
#include <ctime>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

#define PRINT_ALL_TIME 1

//test

const int BUFFER_SIZE = 16;//win        30fsp vs 300fps --- 10buffersize at least
//const int BUFFER_SIZE = 20;//nano
const int show_buffer = 50;// to be large enough


struct bufferItem
{
    cv::Mat frame;              
    std::vector<Yolo::Detection> bboxs; 
};


std::queue<cv::Mat> stage_1_frame;
std::queue<bufferItem> stage_2_buffer;
std::queue<cv::Mat> stage_3_frame;

std::mutex stage_1_mutex;
std::mutex stage_2_mutex;
std::mutex stage_3_mutex;

std::condition_variable stage_1_not_full;
std::condition_variable stage_2_not_full;
std::condition_variable stage_3_not_full;

std::condition_variable stage_1_not_empty;
std::condition_variable stage_2_not_empty;
std::condition_variable stage_3_not_empty;

#define DEVICE 0  
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 

static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int inputIndex = 0;
static const int outputIndex = 1;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

// std::string coco_names[] = {
//     "person:0", "bicycle:1", "car:2", "motorcycle:3", "airplane:4",
//     "bus:5", "train:6", "truck:7", "boat:8", "traffic light:9",
//     "fire hydrant:10", "stop sign:11", "parking meter:12", "bench:13", "bird:14",
//     "cat:15", "dog:16", "horse:17", "sheep:18", "cow:19",
//     "elephant:20", "bear:21", "zebra:22", "giraffe:23", "backpack:24",
//     "umbrella:25", "handbag:26", "tie:27", "suitcase:28", "frisbee:29",
//     "skis:30", "snowboard:31", "sports ball:32", "kite:33", "baseball bat:34",
//     "baseball glove:35", "skateboard:36", "surfboard:37", "tennis racket:38", "bottle:39",
//     "wine glass:40", "cup:41", "fork:42", "knife:43", "spoon:44",
//     "bowl:45", "banana:46", "apple:47", "sandwich:48", "orange:49",
//     "broccoli:50", "carrot:51", "hot dog:52", "pizza:53", "donut:54",
//     "cake:55", "chair:56", "couch:57", "potted plant:58", "bed:59",
//     "dining table:60", "toilet:61", "tv:62", "laptop:63", "mouse:64",
//     "remote:65", "keyboard:66", "cell phone:67", "microwave:68", "oven:69",
//     "toaster:70", "sink:71", "refrigerator:72", "book:73", "clock:74",
//     "vase:75", "scissors:76", "teddy bear:77", "hair drier:78", "toothbrush:79"
// };


class MTDet
{
public:
    ~MTDet()
    {
        std::cout << "MTDet Destructor" << std::endl;
        cudaStreamDestroy(stream);
        CUDA_CHECK(cudaFree(img_device));
        CUDA_CHECK(cudaFreeHost(img_host));
        CUDA_CHECK(cudaFree(buffers[inputIndex]));
        CUDA_CHECK(cudaFree(buffers[outputIndex]));
        cap.release();
        context->destroy();
        engine->destroy();
    }

    MTDet(const std::string &engine_name, const std::string &input_video_path)
    {

        std::ifstream file(engine_name, std::ios::binary);
        if (!file.good()) {
            std::cerr << "read " << engine_name << " error!" << std::endl;
        }
        char* trtModelStream = nullptr;
        size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();

        IRuntime* runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtModelStream, size));
        assert(engine != nullptr);
        context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
        assert(context != nullptr);
        delete[] trtModelStream;

        assert(engine->getNbBindings() == 2);

        CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
        CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));

        if (input_video_path == "rtsp")
        {
            auto rtsp = "rtsp://192.168.0.210:8557/PSIA/Streaming/channels/2?videoCodecType=H.264";
            std::cout << "Input source : rtsp " <<"\n" << std::endl;
            cap = cv::VideoCapture(rtsp, cv::CAP_FFMPEG);
            //if on jetson,see the file:opencv_gst_c++.cpp
        }
        else
        {
            std::cout << "Input source : video file" <<"\n" << std::endl;
            cap = cv::VideoCapture(input_video_path);
        }
        frameSize_ = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        video_fps_ = cap.get(cv::CAP_PROP_FPS);
        std::cout << "video_width: " << frameSize_.width << " video_height: " << frameSize_.height << " fps: " << video_fps_ <<"\n"<< std::endl;
    };

    void readFrame()
    {
        std::cout << "Thread 1 start" << std::endl;

        cv::Mat frame;
        int frame_counter = 0;   //for test
        while (cap.isOpened() && frame_counter < 1000)
        {
            auto start_1 = std::chrono::system_clock::now();
            cap >> frame;
            if(file_processed_done || frame.empty())
            {
                std::cout << "Process Done or Interrupted" << std::endl;
                file_processed_done = true;
                break;
            }

            auto end_1 = std::chrono::system_clock::now();
            rtsp_pull_time.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end_1 - start_1).count());

            std::unique_lock<std::mutex> lock(stage_1_mutex);
            stage_1_not_full.wait(lock, []
                                  { return stage_1_frame.size() < BUFFER_SIZE; });
            stage_1_frame.push(frame);
            stage_1_not_empty.notify_one();

            frame_counter++;
        }
        file_processed_done = true; 
        stage_1_not_empty.notify_all(); 
    }

    void inference()
    {
        std::cout << "Thread 2 start" << std::endl;

        cv::Mat frame;
        static float prob[BATCH_SIZE * OUTPUT_SIZE];
        while (true)
        {
            if (file_processed_done && stage_1_frame.empty())
            {
                std::cout << "Thread 2 break" << std::endl;
                break;
            }

            {
                std::unique_lock<std::mutex> lock(stage_1_mutex);
                stage_1_not_empty.wait(lock, []
                                       { return !stage_1_frame.empty(); });
                frame = stage_1_frame.front(); 
                stage_1_frame.pop();
                stage_1_not_full.notify_one();
            }

            cv::Mat img = frame;

            auto start_2 = std::chrono::system_clock::now();

            std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);
            float* buffer_idx = (float*)buffers[inputIndex];
            imgs_buffer[0] = img;
            size_t  size_image = img.cols * img.rows * 3;
            size_t  size_image_dst = INPUT_H * INPUT_W * 3;
            memcpy(img_host, img.data, size_image);
            CUDA_CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
            auto start_pre = std::chrono::system_clock::now();
            preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);
            auto end_pre = std::chrono::system_clock::now();

            preprocess_time.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - start_pre).count());

            buffer_idx += size_image_dst;

            //run inference
            auto start_infer = std::chrono::system_clock::now();
            doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
            auto end_infer= std::chrono::system_clock::now();

            inference_time.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end_infer - start_infer).count());

            std::vector<std::vector<Yolo::Detection>> batch_res(1); 

            auto& res = batch_res[0];
            
            auto start_nms = std::chrono::system_clock::now();
            nms(res, &prob[0], CONF_THRESH, NMS_THRESH); 
            auto end_nms = std::chrono::system_clock::now();

            nms_time.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end_nms - start_nms).count());

            auto& res_pre = batch_res[0]; 

            bufferItem item;
            item.frame = frame.clone();
            item.bboxs = res_pre;

            auto end_2 = std::chrono::system_clock::now();
            detect_time.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end_2 - start_2).count());

            {
                std::unique_lock<std::mutex> lock2(stage_2_mutex);
                stage_2_not_full.wait(lock2, []
                                      { return stage_2_buffer.size() < BUFFER_SIZE; });
                stage_2_buffer.push(item);
                stage_2_not_empty.notify_one();
            }
        }
    }


    void postprocess()
    {
        std::cout << "Thread 3 start" << std::endl;

        auto start_now = std::chrono::system_clock::now();
        std::time_t now_1 = std::chrono::system_clock::to_time_t(start_now);
        std::tm* now_tm = std::localtime(&now_1);
        std::stringstream sss;
        sss << std::put_time(now_tm, "xxx\\results\\videos\\%Y_%m_%d_%H_%M.avi");
        std::string output_video_name = sss.str();
        cv::VideoWriter writer(output_video_name, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), video_fps_, frameSize_);

        // std::string output_video_name = "G:\\Desktop_temp\\yolov5-tensorrtx-VideoCapture-master\\results\\videos\\sample_detected.mp4";
        // cv::VideoWriter writer(output_video_name, cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, cv::Size(video_width, video_height));

        bufferItem item;
        int framecounter = 0;
        auto start_time = std::chrono::system_clock::now();
        std::string fps_allprocess = "null";
        while (true)
        {
            if (file_processed_done && stage_2_buffer.empty())
            {
                std::cout << "Thread 3 break" << std::endl;
                writer.release();
                break;
            }

            {
                std::unique_lock<std::mutex> lock(stage_2_mutex);
                stage_2_not_empty.wait(lock, []
                                       { return !stage_2_buffer.empty(); });
                item = stage_2_buffer.front();
                stage_2_buffer.pop();
                stage_2_not_full.notify_one();
            }

            auto start_3 = std::chrono::system_clock::now();
            framecounter = (framecounter >= 20) ? 0 : framecounter + 1;
            if (item.bboxs.size() >0) {
                auto in_time_t = std::chrono::system_clock::to_time_t(start_3);
                std::stringstream label_temp;
                label_temp << std::put_time(std::localtime(&in_time_t), "%Y_%m_%d_%H_%M_%S");
                std::string timestamp = label_temp.str() + "_" + std::to_string(framecounter);
                std::string timestamped_label_name = "xxx\\results\\time_labels\\" + timestamp + ".txt";
                std::ofstream timestampedOutFile(timestamped_label_name);
                if (!timestampedOutFile.is_open()) {
                    std::cerr << "Error: Unable to open file: " << timestamped_label_name << std::endl;
                }

                for (size_t j = 0; j < item.bboxs.size(); j++)
                {

                    cv::Rect r = get_rect(item.frame, item.bboxs[j].bbox);
                    float a = (r.x + r.width/2) / float(item.frame.cols);
                    float b = (r.y + r.height/2) / float(item.frame.rows);
                    float c = r.width / float(item.frame.cols);
                    float d = r.height / float(item.frame.rows);                  
                    cv::rectangle(item.frame, r, cv::Scalar(0, 0, 255), 2);
                    cv::putText(item.frame, std::to_string((int)item.bboxs[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 255, 255), 2);
                    timestampedOutFile << item.bboxs[j].class_id << " " << a << " " << b << " " << c << " " << d << "\n";
                }
                timestampedOutFile.close();
            }   
            auto end_3= std::chrono::system_clock::now();
            postprocess_time.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end_3 - start_3).count());
            auto process_time_all = rtsp_pull_time[counter] + detect_time[counter] + postprocess_time[counter];

            // std::stringstream stream_fps;
            // stream_fps << std::fixed << std::setprecision(2) << 1000.f / process_time_all;
            // std::string fps_allprocess = stream_fps.str();
            // cv::putText(item.frame, "FPS: " + fps_allprocess, cv::Point(item.frame.cols * 0.02, item.frame.rows * 0.05), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2, 8);

            if((counter+1) % 30 == 0){
                std::stringstream stream_fps;
                auto frame30_cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_3 - start_time).count();
                stream_fps << std::fixed << std::setprecision(2) << 1000.f / (frame30_cost_time / 30.f);
                fps_allprocess = stream_fps.str();
                std::cout << "FPS: " << fps_allprocess <<std::endl;
                start_time = std::chrono::system_clock::now();
            }

            cv::putText(item.frame, "FPS: " + fps_allprocess, cv::Point(item.frame.cols * 0.3, item.frame.rows * 0.05), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2, 8);
            
#if PRINT_ALL_TIME
            std::cout << "Framecounter: " << counter << std::endl;
            std::cout << "process_time_all:" << process_time_all << "ms" << "  ---  "<<"rtsp_pull_time: "<<rtsp_pull_time[counter] << "ms -- "<<"detect_time: "<<detect_time[counter]<<"ms -- "<<"postprocess_time: "<<postprocess_time[counter]<<"ms"<<std::endl;
            std::cout << "detect_time_detail: " << "pre:" <<preprocess_time[counter] <<"ms --- " <<"infer: " <<inference_time[counter]<<"ms --- "<<"nms: "<<nms_time[counter]<<"ms"<<std::endl;
            // std::cout << "FPS: " << fps_allprocess <<"\n"<< std::endl;
#endif
        
            writer.write(item.frame);
            counter++;

            {
                std::unique_lock<std::mutex> lock3(stage_3_mutex);
                stage_3_not_full.wait(lock3, []
                                      { return stage_3_frame.size() < show_buffer; });
                stage_3_frame.push(item.frame);
                stage_3_not_empty.notify_one();
            }
        }
    }


    void videoshow()
    {
        cv::Mat frame;
        std::cout << "Thread 4 start" << "\n" << std::endl;
        while(true)
        {
            if (file_processed_done && stage_3_frame.empty())
            {
                std::cout << "Thread 4 break" << std::endl;
                break;
            }

            {
                std::unique_lock<std::mutex> lock(stage_3_mutex);
                stage_3_not_empty.wait(lock, []
                                       { return !stage_3_frame.empty(); });
                frame = stage_3_frame.front();
                stage_3_frame.pop();
                stage_3_not_full.notify_one();
            }
            cv::imshow("Inference", frame);
            if (cv::waitKey(1) == 27)  //esc interrupt
            {
                std::cout << "Keyboard Interrupt"<<'\n'<<"Thread 4 break" << std::endl;
                file_processed_done = true;
                //break;  //fatal:!!!  if break here,thread 1 and 2 will not be interrupted and will be hang on
            }
        }
    cv::destroyAllWindows();
    file_processed_done = true;
    }


private:
    std::string input_video_path = "xxx\\sample.mp4"; 
    cv::VideoCapture cap;                 
    double video_fps_;                     
    cv::Size frameSize_;    
    int counter = 0;      

    std::shared_ptr<nvinfer1::ICudaEngine> engine;      
    std::shared_ptr<nvinfer1::IExecutionContext> context; 

    bool file_processed_done = false; 
     
    std::vector<double> rtsp_pull_time;
    std::vector<double> detect_time;
    std::vector<double> postprocess_time;
    std::vector<double> preprocess_time;
    std::vector<double> inference_time;
    std::vector<double> nms_time;

    cudaStream_t stream;
    float* buffers[2];
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;

    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize) {
        context.enqueue(batchSize, buffers, stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }
};


int main(int argc, char **argv)
{
    cudaSetDevice(DEVICE);

    if (argc < 3)
    {
        std::cerr << "correct usage: " << argv[0] << " <engine_name> <input_video_path>  !" << std::endl;
        return -1;
    }
    auto engine_name = argv[1];             
    std::string input_video_path = argv[2];    

    auto det = MTDet(engine_name, input_video_path);
    std::thread T_readFrame(&MTDet::readFrame, &det);
    std::thread T_inference(&MTDet::inference, &det);
    std::thread T_postprocess(&MTDet::postprocess, &det);
    std::thread T_videoshow(&MTDet::videoshow, &det);

    T_readFrame.join();
    T_inference.join();
    T_postprocess.join();
    T_videoshow.join();

    return 0;
}
