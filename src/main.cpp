#include <iostream>
#include <time.h>

#include "NetWork.h"

#include "dataLoader.h"
#include "Utils.h"

/** indicate 0 ~ 9 */
#define NUM_NET_OUT 10
#define NUM_HIDDEN 100
#define NET_LEARNING_RATE 0.5

#define TRAIN_IMAGES_URL "../data/train-images.idx3-ubyte"
#define TRAIN_LABELS_URL "../data/train-labels.idx1-ubyte"

#define TEST_IMANGES_URL "../data/t10k-images.idx3-ubyte"
#define TEST_LABELS_URL  "../data/t10k-labels.idx1-ubyte"


void showNumber(unsigned char pic[], int width, int height) {
    int idx = 0;
    for (int i=0; i < height; i++) {
        for (int j = 0; j < width; j++ ) {

            if (pic[idx++]) {
                cout << "1";
            } else {
                cout << " ";
            }
        }
        cout << endl;
    }
}

inline void preProcessInputData(const unsigned char src[], double out[], int size) {
    for (int i = 0; i < size; i++) {
        out[i] = (src[i] >= 128) ? 1.0 : 0.0;
    }
}

inline void preProcessInputDataWithNoise(const unsigned char src[], double out[], int size) {
    for (int i = 0; i < size; i++) {
        out[i] = ((src[i] >= 128) ? 1.0 : 0.0) + RandFloat() * 0.1;
    }
}

inline void preProcessInputData(const unsigned char src[],int size, std::vector<int>& indexs) {
    for (int i = 0; i < size; i++) {
        if (src[i] >= 128) {
            indexs.push_back(i);
        }
    }
}

double trainEpoch(dataLoader& src, NetWork& bpnn, int imageSize, int numImages) {
    double net_target[NUM_NET_OUT];
    char* temp = new char[imageSize];

    double* net_train = new double[imageSize];
    for (int i = 0; i < numImages; i++) {
        int label = 0;
        memset(net_target, 0, NUM_NET_OUT * sizeof(double));

        if (src.read(&label, temp)) {
            net_target[label] = 1.0;
            preProcessInputData((unsigned char*)temp, net_train, imageSize);
            bpnn.training(net_train, net_target);
        }
        else {
            cout << "读取训练数据失败" << endl;
            break;
        }
        cout << "已学习：" << i << "\r";
    }

    // cout << "the error is:" << bpnn.getError() << " after training " << endl;

    delete []net_train;
    delete []temp;

    return bpnn.getError();
}

int testRecognition(dataLoader& testData, NetWork& bpnn, int imageSize, int numImages) {
    int ok_cnt = 0;
    double* net_out = NULL;
    char* temp = new char[imageSize];
    double* net_test = new double[imageSize];
    for (int i = 0; i < numImages; i++) {
        int label = 0;

        if (testData.read(&label, temp)) {			
            preProcessInputData((unsigned char*)temp, net_test, imageSize);
            bpnn.process(net_test, &net_out);

            int idx = -1;
            double max_value = -99999;
            for (int i = 0; i < NUM_NET_OUT; i++) {
                if (net_out[i] > max_value) {
                    max_value = net_out[i];
                    idx = i;
                }
            }

            if (idx == label) {
                ok_cnt++;
            }

        }
        else {
            cout << "read test data failed" << endl;
            break;
        }
    }
    delete []net_test;
    delete []temp;
    return ok_cnt;
}


int main(int argc, char* argv[]) {
    dataLoader src;
    dataLoader testData;
    NetWork* bpnn = NULL;
    srand((int)time(0));

    if (src.openImageFile(TRAIN_IMAGES_URL) && src.openLabelFile(TRAIN_LABELS_URL)) {
        int imageSize = src.imageLength();
        int numImages = src.numImage();
        int epochMax = 1;
        double expectErr = 0.1;
        bpnn = new NetWork(imageSize, NET_LEARNING_RATE);
        // 加入隐藏层
        bpnn->addNeuronLayer(NUM_HIDDEN);
        // 加入输出层
        bpnn->addNeuronLayer(NUM_NET_OUT);

        cout << "开始进行训练：" << endl;
        uint64_t st = timeNowMs();

        for (int i = 0; i < epochMax; i++) {
            double err = trainEpoch(src, *bpnn, imageSize, numImages);
            src.reset();
        }

        cout << "训练结束，花费时间: " << (timeNowMs() - st)/1000 << "秒" << endl;

        st = timeNowMs();
        
        if (testData.openImageFile(TEST_IMANGES_URL) && testData.openLabelFile(TEST_LABELS_URL)) {
            imageSize = testData.imageLength();
            numImages = testData.numImage();
            
            cout << "开始进行测试：" << endl;

            int ok_cnt = testRecognition(testData, *bpnn, imageSize, numImages);

            cout << "测试结束，花费时间："
                << (timeNowMs() - st)/1000 << "秒, " 
                <<  "成功比例: " << ok_cnt/(double)numImages*100 << "%" << endl;
        }
        else {
            cout << "打开测试文件失败" << endl;
        }


    }
    else {
        cout << "open train image file failed" << endl;
    }

    if (bpnn) {
        delete bpnn;
    }

    getchar();

    return 0;
}