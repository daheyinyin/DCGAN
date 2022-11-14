
#include <iostream>
#include <vector>
#include "DCGAN.h"
#include "MxBase/Log/Log.h"

const uint32_t IMG_CARDINALITY = 1000;
std::vector<double> g_inferCost;

void InitDCGANParam(InitParam *initParam, char* argv[]) {
    initParam->deviceId = 0;
    initParam->checkTensor = true;
    initParam->modelPath = argv[1];
    initParam->savePath = argv[2];
    initParam->imageNum = std::stoi(argv[3]);
    initParam->imageWidth = 32;
    initParam->imageHeight = 32;
    initParam->batchSize = 16;
}

int main(int argc, char* argv[]) {
    InitParam initParam;
    InitDCGANParam(&initParam, argv);

    // Create dcgan
    auto dcgan = std::make_shared<DCGAN>();

    // do init
    APP_ERROR ret = dcgan->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "DCGAN init failed, ret=" << ret << ".";
        return ret;
    }
    LogInfo << "End to Init DCGAN.";

    for (uint32_t i = 0; i < initParam.imageNum; i++) {
        // do process
        LogInfo << "generate image " << i;
        ret = dcgan->Process(i);
        if (ret != APP_ERR_OK) {
            LogError << "DCGAN process failed, ret=" << ret << ".";
            dcgan->DeInit();
            return ret;
        }
    }

    // do deinit
    dcgan->DeInit();

    double costSum = 0;
    for (uint32_t i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * IMG_CARDINALITY / costSum << " images/sec.";
    return APP_ERR_OK;
}
