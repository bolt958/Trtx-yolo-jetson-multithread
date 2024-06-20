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


#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !

// stuff we know about the network and the input/output blobs
//Yolo结构体类定义在yololayer.h中，其中的Detection结构体定义了检测框的信息，包括bbox坐标、置信度、类别id
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

static int get_width(int x, float gw, int divisor = 8) {
    return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}

ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    /* ------ yolov5 backbone------ */
    auto conv0 = convBlock(network, weightMap, *data,  get_width(64, gw), 6, 2, 1,  "model.0");
    assert(conv0);
    auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto bottleneck_csp8 = C3(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto spp9 = SPPF(network, weightMap, *bottleneck_csp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, "model.9");
    /* ------ yolov5 head ------ */
    auto conv10 = convBlock(network, weightMap, *spp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

    auto upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(ResizeMode::kNEAREST);
    upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

    ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

    auto upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(ResizeMode::kNEAREST);
    upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

    ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

ICudaEngine* build_engine_p6(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);
    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);
    
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    /* ------ yolov5 backbone------ */
    auto conv0 = convBlock(network, weightMap, *data,  get_width(64, gw), 6, 2, 1,  "model.0");
    auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto c3_2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *c3_2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto c3_4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *c3_4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto c3_6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *c3_6->getOutput(0), get_width(768, gw), 3, 2, 1, "model.7");
    auto c3_8 = C3(network, weightMap, *conv7->getOutput(0), get_width(768, gw), get_width(768, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto conv9 = convBlock(network, weightMap, *c3_8->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.9");
    auto c3_10 = C3(network, weightMap, *conv9->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), true, 1, 0.5, "model.10");
    auto sppf11 = SPPF(network, weightMap, *c3_10->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, "model.11");

    /* ------ yolov5 head ------ */
    auto conv12 = convBlock(network, weightMap, *sppf11->getOutput(0), get_width(768, gw), 1, 1, 1, "model.12");
    auto upsample13 = network->addResize(*conv12->getOutput(0));
    assert(upsample13);
    upsample13->setResizeMode(ResizeMode::kNEAREST);
    upsample13->setOutputDimensions(c3_8->getOutput(0)->getDimensions());
    ITensor* inputTensors14[] = { upsample13->getOutput(0), c3_8->getOutput(0) };
    auto cat14 = network->addConcatenation(inputTensors14, 2);
    auto c3_15 = C3(network, weightMap, *cat14->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.15");

    auto conv16 = convBlock(network, weightMap, *c3_15->getOutput(0), get_width(512, gw), 1, 1, 1, "model.16");
    auto upsample17 = network->addResize(*conv16->getOutput(0));
    assert(upsample17);
    upsample17->setResizeMode(ResizeMode::kNEAREST);
    upsample17->setOutputDimensions(c3_6->getOutput(0)->getDimensions());
    ITensor* inputTensors18[] = { upsample17->getOutput(0), c3_6->getOutput(0) };
    auto cat18 = network->addConcatenation(inputTensors18, 2);
    auto c3_19 = C3(network, weightMap, *cat18->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.19");

    auto conv20 = convBlock(network, weightMap, *c3_19->getOutput(0), get_width(256, gw), 1, 1, 1, "model.20");
    auto upsample21 = network->addResize(*conv20->getOutput(0));
    assert(upsample21);
    upsample21->setResizeMode(ResizeMode::kNEAREST);
    upsample21->setOutputDimensions(c3_4->getOutput(0)->getDimensions());
    ITensor* inputTensors21[] = { upsample21->getOutput(0), c3_4->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors21, 2);
    auto c3_23 = C3(network, weightMap, *cat22->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.23");

    auto conv24 = convBlock(network, weightMap, *c3_23->getOutput(0), get_width(256, gw), 3, 2, 1, "model.24");
    ITensor* inputTensors25[] = { conv24->getOutput(0), conv20->getOutput(0) };
    auto cat25 = network->addConcatenation(inputTensors25, 2);
    auto c3_26 = C3(network, weightMap, *cat25->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.26");

    auto conv27 = convBlock(network, weightMap, *c3_26->getOutput(0), get_width(512, gw), 3, 2, 1, "model.27");
    ITensor* inputTensors28[] = { conv27->getOutput(0), conv16->getOutput(0) };
    auto cat28 = network->addConcatenation(inputTensors28, 2);
    auto c3_29 = C3(network, weightMap, *cat28->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.29");

    auto conv30 = convBlock(network, weightMap, *c3_29->getOutput(0), get_width(768, gw), 3, 2, 1, "model.30");
    ITensor* inputTensors31[] = { conv30->getOutput(0), conv12->getOutput(0) };
    auto cat31 = network->addConcatenation(inputTensors31, 2);
    auto c3_32 = C3(network, weightMap, *cat31->getOutput(0), get_width(2048, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.32");

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*c3_23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.0.weight"], weightMap["model.33.m.0.bias"]);
    IConvolutionLayer* det1 = network->addConvolutionNd(*c3_26->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.1.weight"], weightMap["model.33.m.1.bias"]);
    IConvolutionLayer* det2 = network->addConvolutionNd(*c3_29->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.2.weight"], weightMap["model.33.m.2.bias"]);
    IConvolutionLayer* det3 = network->addConvolutionNd(*c3_32->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.3.weight"], weightMap["model.33.m.3.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, "model.33", std::vector<IConvolutionLayer*>{det0, det1, det2, det3});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, bool& is_p6, float& gd, float& gw, std::string& wts_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = nullptr;
    if (is_p6) {
        engine = build_engine_p6(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    } else {
        engine = build_engine(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    }
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
    //序列化引擎：保存为engine文件，准备阶段，不会在推理时使用---wts转engine
}



void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize) {
    // infer on the batch asynchronously, and DMA output back to host
    //DMA: Direct Memory Access直接内存访问，GPU直接将数据从设备内存拷贝到主机内存，而不需要CPU参与
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    // 在doInference函数中，context.enqueue阶段使用buffers[0]中的数据进行模型推理
    // 然后将推理结果保存到buffers[1]中。随后，使用cudaMemcpyAsync函数将buffers[1]中的数据异步拷贝到output数组中
}



bool parse_args(std::string argv[], std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw) {  
    if (std::string(argv[1]) == "-s" ) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto net = std::string(argv[4]);
        if (net[0] == 'n') {
            gd = 0.33;
            gw = 0.25;
        } else if (net[0] == 's') {
            gd = 0.33;
            gw = 0.50;
        } else if (net[0] == 'm') {
            gd = 0.67;
            gw = 0.75;
        } else if (net[0] == 'l') {
            gd = 1.0;
            gw = 1.0;
        } else if (net[0] == 'x') {
            gd = 1.33;
            gw = 1.25;
        } else if (net[0] == 'c' ) {
            gd = atof(argv[5].c_str());
            gw = atof(argv[6].c_str());
        } else {
            return false;
        }
        if (net.size() == 2 && net[1] == '6') {
            is_p6 = true;
        }
    } else if (std::string(argv[1]) == "-d" ) {
        engine = std::string(argv[3]);
       // img_dir = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}



int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);

    std::string wts_name = "";
    std::string engine_name = "";
    bool is_p6 = false;
    float gd = 0.0f, gw = 0.0f;
    //std::string img_dir;
    //std::string class_names[] = {"AUV_1", "AUV_2", "HOV_1","HOV_2","UUV_unknown","test"};

    std::string parse[10];
    parse[0] = "./yolov5"; 
    parse[1] = "-d"; 
    parse[2] = "yolov5n-6.0.wts"; 
    parse[3] = "yolov5n-6.0.engine"; 
    parse[4] = "n"; 
    if (!parse_args(parse, wts_name, engine_name, is_p6, gd, gw)) {
        std::cerr << "arguments not right!" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    // 如果没有engine文件，就根据wts文件创建engine文件，如果有engine文件，就直接读取engine文件
    if (!wts_name.empty()) {
        
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream, is_p6, gd, gw, wts_name);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }





    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char* trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    //加载的二进制模型文件最终保存到了trtModelStream指针指向的内存区域中
    file.close();

    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    float* buffers[2];
    //buffer[0]存储输入数据的GPU内存地址，即经过预处理的图像数据，大小为BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)
    //buffer[1]存储输出数据，即推理结果
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    //上述代码主要做了：1.反序列化引擎 2.创建推理上下文 3.创建输入输出缓冲区buffer 4.创建CUDA流 5.准备输入数据缓存



    cv::VideoCapture capture("black.mp4");
    //cv::VideoCapture capture(0);
    int video_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int video_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = capture.get(cv::CAP_PROP_FPS);


    //name the output video with the current time---precise to minute
    auto start_now = std::chrono::system_clock::now();
    std::time_t now_1 = std::chrono::system_clock::to_time_t(start_now);
    std::tm* now_tm = std::localtime(&now_1);
    std::stringstream sss;
    sss << std::put_time(now_tm, "G:\\Desktop_temp\\yolov5-tensorrtx-VideoCapture-master\\results\\videos\\%Y_%m_%d_%H_%M.avi");
    std::string output_video_name = sss.str();
    cv::VideoWriter writer(output_video_name, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(video_width, video_height));

    // std::string output_video_name = "G:\\Desktop_temp\\yolov5-tensorrtx-VideoCapture-master\\results\\videos\\test.avi";
    // cv::VideoWriter writer(output_video_name, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(video_width, video_height));
    // std::string output_video_name = "G:\\Desktop_temp\\yolov5-tensorrtx-VideoCapture-master\\results\\videos\\sample_detected.mp4";
    // cv::VideoWriter writer(output_video_name, cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, cv::Size(video_width, video_height));
  
    cv::Mat frame;
    int framecounter = 0;

    while (cv::waitKey(1) != 27)
    {

        framecounter++;

        capture >> frame;
        cv::Mat img = frame;
        if (img.empty())
        {
            std::cout << "No Video input\n" << std::endl;
            return -1;
        }

        //here start the whole process
        auto start_process = std::chrono::system_clock::now();

        std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);
        float* buffer_idx = (float*)buffers[inputIndex];
        //buffers是一个指向GPU内存的指针数组，inputIndex是输入数据（图像）的索引。buffer_idx是一个指向浮点数的指针，指向模型输入数据的GPU内存地址
        imgs_buffer[0] = img;
        //输入图像img图像帧存储到imgs_buffer的第一个位置。这表明在这个批次中，只处理一个图像。
        size_t  size_image = img.cols * img.rows * 3;
        size_t  size_image_dst = INPUT_H * INPUT_W * 3;
        //copy data to pinned memory
        memcpy(img_host, img.data, size_image);
        //将图像数据从img（cv::Mat对象）拷贝到主机的锁页内存（img_host）中。锁页内存是一种可以防止操作系统将其交换到磁盘的内存，这样可以加速从主机到GPU的数据传输。
        //copy data to device memory
        CUDA_CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
        //异步将图像数据从主机的锁页内存（img_host--CPU）拷贝到GPU的设备内存（img_device）中。这里使用了CUDA流（stream），以便在不阻塞主线程的情况下执行数据传输。
        preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);
        //处理后的图像数据存储在模型的输入缓冲区（buffer_idx指向的位置），应该就是buffers[inputIndex]中
        buffer_idx += size_image_dst;
        //更新buffer_idx指针，使其指向下一个图像数据应该存储的位置。在这个例子中，由于只处理一个图像，这一步实际上是多余的。

        //run inference
        auto start_infer = std::chrono::system_clock::now();
        doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);//检测结果保存在buffer里，再通过prob传出
        auto end_infer= std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_infer - start_infer).count() << "ms" << std::endl;
        std::vector<std::vector<Yolo::Detection>> batch_res(1); //定义在yololayer.h中的一个结构体，用于存储检测结果
        //两层嵌套的意义：第一层是batch，第二层是每个batch中的检测结果，不过这里只有一个batch，所以只有一层，所以是（1）

        auto& res = batch_res[0];//[0]代表第一个批次的检测结果，而不是单单指bbox
        nms(res, &prob[0], CONF_THRESH, NMS_THRESH);//NMS非极大值抑制，去除重叠的框  nms定义在common.cpp中，输入prob[0]，处理后的数据存储在res中

        auto& res_pre = batch_res[0]; //由于&res是一个引用，所以res改变后batch_res也会改变,这里已经是处理后的数据
        cv::Mat img_pre = imgs_buffer[0];//这里的imgs_buffer[0]仍然是原始图像，没有发生改变，因为对图像的处理都在gpu和buffer中进行，没有改变CPU内存数据
        //以便后续在原图上画框以及打印各种信息
        std::cout <<"number of detected objects: " <<res_pre.size() << std::endl;

        
        //res_pre.size() : the number of detected objects in one frame
        if(res_pre.size() > 0) {
            //如果检测到目标
            // name the label.txt file with the current time---year_month_day_hour_minute_second_framecounter.txt
            auto in_time_t = std::chrono::system_clock::to_time_t(start_process);
            std::stringstream label_temp;
            label_temp << std::put_time(std::localtime(&in_time_t), "%Y_%m_%d_%H_%M_%S");
            std::string timestamp = label_temp.str() + "_" + std::to_string(framecounter);
            std::string timestamped_label_name = "G:\\Desktop_temp\\yolov5-tensorrtx-VideoCapture-master\\results\\time_labels\\" + timestamp + ".txt";
            std::ofstream timestampedOutFile(timestamped_label_name);
            if (!timestampedOutFile.is_open()) {
                std::cerr << "Error: Unable to open file: " << timestamped_label_name << std::endl;
                return 1;
            }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
            for (size_t j = 0; j < res_pre.size(); j++) {
                //遍历一帧图像中的所有检测结果
                //Get the coordinates of the detection box (upper left corner coordinates + length and width)
                //Convert to yolo format: center point coordinates + length and width (normalized)
                cv::Rect r = get_rect(img_pre, res_pre[j].bbox);
                float a = (r.x + r.width/2) / float(img.cols);
                float b = (r.y + r.height/2) / float(img.rows);
                float c = r.width / float(img.cols);
                float d = r.height / float(img.rows);
                
                cv::rectangle(img_pre, r, cv::Scalar(0, 0, 255), 1);
                //category
                //cv::putText(img_pre, class_names[(int)res[j].class_id], cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 255, 255), 1);
                cv::putText(img_pre, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 255, 255), 1);
                timestampedOutFile << res[j].class_id << " " << a << " " << b << " " << c << " " << d << "\n";
                  
            }

            timestampedOutFile.close();
        }

        //calculate the total fps every frame
        auto end_process= std::chrono::system_clock::now();
        std::stringstream stream_fps;
        stream_fps << std::fixed << std::setprecision(2) << 1000.0 / (std::chrono::duration_cast<std::chrono::milliseconds>(end_process- start_process).count() + 1);
        std::string fps_allprocess = stream_fps.str();
        cv::putText(frame, "FPS: " + fps_allprocess, cv::Point(img.cols * 0.02, img.rows * 0.05), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2, 8);
        std::cout << "FPS: " << fps_allprocess <<"\n"<< std::endl;
       
        //show the result and save the result as a video
        cv::namedWindow("Inference", cv::WINDOW_FREERATIO);
        cv::imshow("Inference" , img_pre);
        writer.write(frame);

    }
    
    //这些步骤是在推理结束后进行的，用于释放资源
    writer.release();

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFreeHost(img_host));
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}

