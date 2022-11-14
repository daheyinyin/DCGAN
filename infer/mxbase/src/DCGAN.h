

#ifndef MXBASE_DCGAN_H
#define MXBASE_DCGAN_H

#include <dirent.h>

#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <cstdlib>
#include <ctime>
#include <random>

#include <opencv2/opencv.hpp>
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

extern std::vector<double> g_inferCost;
const uint32_t FLOAT32_TYPE_BYTE_NUM = 4;
const float NORMALIZE_MEAN = 127.5;
const float NORMALIZE_STD = 127.5;
const uint32_t CHANNEL = 3;

struct InitParam {
    uint32_t deviceId;
    bool checkTensor;
    std::string modelPath;
    std::string savePath;
    uint32_t imageNum;
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t batchSize;
};

class DCGAN {
 public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(std::vector<MxBase::TensorBase> outputs, std::vector<cv::Mat> &resultMats);
    APP_ERROR Process(uint32_t gen_id);
    APP_ERROR SaveResult(std::vector<cv::Mat> &resultMats, const std::string &imgName);
    APP_ERROR CreateRandomTensorBase(std::vector<MxBase::TensorBase> &inputs);

 private:
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    MxBase::ModelDesc modelDesc_;
    std::mt19937 gen_;
    std::normal_distribution<float> dis_;
};
#endif
