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

#define PRINT_STEP_TIME 1
#define PRINT_ALL_TIME 1

const int BUFFER_SIZE = 10;

struct bufferItem
{
    cv::Mat frame;              
    std::vector<Yolo::Detection> bboxs; 
};

std::queue<cv::Mat> stage_1_frame;
std::queue<bufferItem> stage_2_buffer;

std::mutex stage_1_mutex;
std::mutex stage_2_mutex;

std::condition_variable stage_1_not_full;
std::condition_variable stage_2_not_full;

std::condition_variable stage_1_not_empty;
std::condition_variable stage_2_not_empty;

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


class MTDet
{
public:
    ~MTDet()
    {
        std::cout << "MTDet destructor" << std::endl;
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
        //Runtime* runtime = std::unique_ptr<nvinfer1::IRuntime>(createInferRuntime(gLogger));
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
            std::cout << "input source : rtsp " << std::endl;
            cap = cv::VideoCapture(rtsp, cv::CAP_FFMPEG);
        }
        else
        {
            std::cout << "input source : video file" << std::endl;
            cap = cv::VideoCapture(input_video_path);
        }
        frameSize_ = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        video_fps_ = cap.get(cv::CAP_PROP_FPS);
        std::cout << "video_width: " << frameSize_.width << " video_height: " << frameSize_.height << " fps: " << video_fps_ << std::endl;
    };

    void readFrame()
    {
        std::cout << "thread 1 start" << std::endl;

        cv::Mat frame;
        while (cap.isOpened())
        {
            auto start_1 = std::chrono::system_clock::now();
            cap >> frame;
            if (frame.empty())
            {
                std::cout << "process done" << std::endl;
                file_processed_done = true;
                break;
            }
            auto end_1 = std::chrono::system_clock::now();
            auto rtsp_pull_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_1 - start_1).count();

#if PRINT_STEP_TIME
            std::cout << "rtsp pull time: " << rtsp_pull_time << "ms" << std::endl;
#endif

            std::unique_lock<std::mutex> lock(stage_1_mutex);
            stage_1_not_full.wait(lock, []
                                  { return stage_1_frame.size() < BUFFER_SIZE; });
            stage_1_frame.push(frame);
            stage_1_not_empty.notify_one();
        }
    }

    void inference()
    {
        std::cout << "thread 2 start" << std::endl;

        cv::Mat frame;
        static float prob[BATCH_SIZE * OUTPUT_SIZE];
        while (true)
        {
            if (file_processed_done && stage_1_frame.empty())
            {
                std::cout << "thread 2 break" << std::endl;
                cudaStreamDestroy(stream);
                CUDA_CHECK(cudaFree(img_device));
                CUDA_CHECK(cudaFreeHost(img_host));
                CUDA_CHECK(cudaFree(buffers[inputIndex]));
                CUDA_CHECK(cudaFree(buffers[outputIndex]));
                context->destroy();
                engine->destroy();
                //runtime->destroy();
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
            cv::Mat img = frame.clone();
            //cv::Mat img = frame;

            auto start_2 = std::chrono::system_clock::now();

            std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);
            float* buffer_idx = (float*)buffers[inputIndex];
            imgs_buffer[0] = img;
            size_t  size_image = img.cols * img.rows * 3;
            size_t  size_image_dst = INPUT_H * INPUT_W * 3;
            memcpy(img_host, img.data, size_image);
            CUDA_CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
            preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);
            buffer_idx += size_image_dst;

            //run inference
            auto start_infer = std::chrono::system_clock::now();
            doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
            auto end_infer= std::chrono::system_clock::now();

#if PRINT_STEP_TIME
            std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_infer - start_infer).count() << "ms" << std::endl;
#endif

            std::vector<std::vector<Yolo::Detection>> batch_res(1); 

            auto& res = batch_res[0];
            nms(res, &prob[0], CONF_THRESH, NMS_THRESH); 

            auto& res_pre = batch_res[0]; 
            //cv::Mat img_pre = imgs_buffer[0];

            bufferItem item;
            item.frame = frame.clone();//深拷贝，不会影响原图，相互独立
            // item.frame = frame;
            item.bboxs = res_pre;

            auto end_2 = std::chrono::system_clock::now();
            detect_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_2 - start_2).count();

#if PRINT_STEP_TIME
            std::cout << "detect time: " << detect_time << "ms" << std::endl;
#endif

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
        std::cout << "thread 3 start" << std::endl;

        auto start_now = std::chrono::system_clock::now();
        std::time_t now_1 = std::chrono::system_clock::to_time_t(start_now);
        std::tm* now_tm = std::localtime(&now_1);
        std::stringstream sss;
        sss << std::put_time(now_tm, "G:\\Desktop_temp\\yolov5-tensorrtx-VideoCapture-master\\results\\videos\\%Y_%m_%d_%H_%M.avi");
        std::string output_video_name = sss.str();
        cv::VideoWriter writer(output_video_name, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), video_fps_, frameSize_);

        bufferItem item;
        //cv::Mat frame;
        while (true)
        {
            if (file_processed_done && stage_2_buffer.empty())
            {
                std::cout << "thread 3 break" << std::endl;
                writer.release();
                break;
            }

            {
                std::unique_lock<std::mutex> lock(stage_2_mutex);
                stage_2_not_empty.wait(lock, []
                                       { return !stage_2_buffer.empty(); });
                item = stage_2_buffer.front();
                // frame = item.frame.clone();
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
                std::string timestamped_label_name = "G:\\Desktop_temp\\yolov5-tensorrtx-VideoCapture-master\\results\\time_labels\\" + timestamp + ".txt";
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
                    cv::rectangle(item.frame, r, cv::Scalar(0, 0, 255), 1);
                    cv::putText(item.frame, std::to_string((int)item.bboxs[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 255, 255), 1);
                    timestampedOutFile << item.bboxs[j].class_id << " " << a << " " << b << " " << c << " " << d << "\n";
                }
                timestampedOutFile.close();
            }   
            auto end_3= std::chrono::system_clock::now();
            auto postprocess_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_3 - start_3).count();

#if PRINT_STEP_TIME
            std::cout << "postprocess time: " << postprocess_time << "ms" << std::endl;
#endif
            auto process_time_all = rtsp_pull_time + detect_time + postprocess_time;

#if PRINT_ALL_TIME
            std::cout << "process_time_all: " << postprocess_time << "ms" << std::endl;
#endif

            std::stringstream stream_fps;
            stream_fps << std::fixed << std::setprecision(2) << 1000.0 / process_time_all;
            std::string fps_allprocess = stream_fps.str();
            cv::putText(item.frame, "FPS: " + fps_allprocess, cv::Point(item.frame.cols * 0.02, item.frame.rows * 0.05), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2, 8);
            
#if PRINT_ALL_TIME
            std::cout << "FPS: " << fps_allprocess <<"\n"<< std::endl;
#endif
        
            //cv::namedWindow("frame", cv::WINDOW_FREERATIO);
            //cv::imshow("frame" , item.frame);
            writer.write(item.frame);
        }
    }


private:
    std::string input_video_path = "G:\\Desktop_temp\\yolov5-tensorrtx-VideoCapture-master\\videos\\sample.mp4"; 
    cv::VideoCapture cap;                 
    double video_fps_;                     
    cv::Size frameSize_;
    int framecounter = 0;            

    std::shared_ptr<nvinfer1::ICudaEngine> engine;      
    std::shared_ptr<nvinfer1::IExecutionContext> context; 

    bool file_processed_done = false; 
    double rtsp_pull_time = 0;        
    double detect_time = 0;             
    double postprocess_time = 0;       

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
        std::cerr << "correct usage: " << argv[0] << " <engine_name> <input_path_path> " << std::endl;
        return -1;
    }
    auto engine_name = argv[1];             
    std::string input_video_path = argv[2];    

    auto det = MTDet(engine_name, input_video_path);
    std::thread T_readFrame(&MTDet::readFrame, &det);
    std::thread T_inference(&MTDet::inference, &det);
    std::thread T_postprocess(&MTDet::postprocess, &det);

    T_readFrame.join();
    T_inference.join();
    T_postprocess.join();

    return 0;
}
