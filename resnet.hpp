#ifndef RESNET_H
#define RESNET_H

#include <torch/torch.h>

torch::nn::Conv2d conv3x3(int64_t inputChannels, int64_t outputChannels, int64_t stride) {
    auto options = torch::nn::Conv2dOptions(inputChannels, outputChannels, /*kernel_size=*/3);
    options = options.stride(stride).padding(1).with_bias(false);
    return std::make_shared<torch::nn::Conv2dImpl>(options);
}

torch::nn::Conv2d conv1x1(int64_t inputChannels, int64_t outputChannels, int64_t stride) {
    auto options = torch::nn::Conv2dOptions(inputChannels, outputChannels, /*kernel_size=*/1);
    options = options.stride(stride).with_bias(false);
    return std::make_shared<torch::nn::Conv2dImpl>(options);
}

struct BasicBlock : torch::nn::Module {
    static const int64_t EXPANSION = 1;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm bn1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm bn2;
    torch::nn::Sequential downsample;

    BasicBlock(int64_t inplanes, int64_t planes, int64_t stride, torch::nn::Sequential downsample)
    :   conv1(conv3x3(inplanes, planes, stride)),
        bn1(torch::nn::BatchNorm(torch::nn::BatchNormOptions(planes))),
        conv2(conv3x3(planes, planes, /*stride*/1)),
        bn2(torch::nn::BatchNorm(torch::nn::BatchNormOptions(planes))),
        downsample(downsample)
    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("downsample", downsample);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto identity = x;

        auto out = conv1->forward(x);
        out = bn1->forward(out);
        out = torch::relu(out);
        out = conv2->forward(out);
        out = bn2->forward(out);

        if (this->downsample.get()->size() > 0) {
            identity = this->downsample->forward(x);
        }

        out = out + identity;
        out = torch::relu(out);
        return out;
    }
};

struct ResNet : torch::nn::Module {
    int64_t inplanes = 64;

    torch::nn::Conv2dOptions conv1Options;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm bn1;
    torch::nn::Sequential layer1;
    torch::nn::Sequential layer2;
    torch::nn::Sequential layer3;
    torch::nn::Sequential layer4;
    torch::nn::Linear fc1;

    ResNet(int64_t inputDepth, int layers[]) 
    :
    conv1Options(torch::nn::Conv2dOptions(inputDepth, 64, /*kernel_size=*/7).stride(2).padding(3).with_bias(false)),
    conv1(std::make_shared<torch::nn::Conv2dImpl>(conv1Options)),
    bn1(std::make_shared<torch::nn::BatchNormImpl>(64)),
    layer1(make_layer_basic(64,  layers[0], /*stride=*/1)),
    layer2(make_layer_basic(128, layers[1], /*stride=*/2)),
    layer3(make_layer_basic(256, layers[2], /*stride=*/2)),
    layer4(make_layer_basic(512, layers[3], /*stride=*/2)),
    fc1(std::make_shared<torch::nn::LinearImpl>(512, 9))
    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("fc1", fc1);

        //Init weights
        // for (auto module_shared : this->modules()) {
        //     auto module = module_shared.get();
        //     if (dynamic_cast<torch::nn::Conv2d*>(module)) {
        //         //TODO: Use Kaiming normal not xavier
        //         auto convLayer = dynamic_cast<torch::nn::Conv2d*>(module)->get();
        //         torch::nn::init::xavier_normal_(convLayer->weight);
        //     }
        //     else if (dynamic_cast<torch::nn::BatchNorm*>(module)) {
        //         auto batchNormLayer = dynamic_cast<torch::nn::BatchNorm*>(module)->get();
        //         torch::nn::init::constant_(batchNormLayer->weight, 1);
        //         torch::nn::init::constant_(batchNormLayer->bias, 0);
        //     }
        // }
    }

    torch::nn::Sequential make_layer_basic(int64_t planes, int64_t blocks, int64_t stride) {

        torch::nn::Sequential downsample = std::make_shared<torch::nn::SequentialImpl>();
        if(stride != 1 || this->inplanes != planes * BasicBlock::EXPANSION) { 
            downsample = torch::nn::Sequential(
                std::make_shared<torch::nn::SequentialImpl>(
                conv1x1(this->inplanes, planes * BasicBlock::EXPANSION, stride),
                torch::nn::BatchNorm(planes * BasicBlock::EXPANSION))
            );
        }

        torch::nn::Sequential layers = std::make_shared<torch::nn::SequentialImpl>();
        auto newBlock = std::make_shared<BasicBlock>(this->inplanes, planes, stride, downsample);
        layers->push_back(newBlock);
        this->inplanes = planes * BasicBlock::EXPANSION;

        for(int64_t i = 0; i < blocks; i++) {
            torch::nn::Sequential empty_downsample = std::make_shared<torch::nn::SequentialImpl>();
            newBlock = std::make_shared<BasicBlock>(this->inplanes, planes, /*stride=*/1, empty_downsample);
            layers->push_back(newBlock);
        }

        return layers;
    }

    torch::Tensor forward(torch::Tensor x) {
        x = this->conv1->forward(x);
        x = this->bn1->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, /*kernel_size*/{3}, /*stride*/{2}, /*padding*/{1});

        x = this->layer1->forward(x);
        x = this->layer2->forward(x);
        x = this->layer3->forward(x);
        x = this->layer4->forward(x);

        x = torch::adaptive_avg_pool2d(x, {1,1});
        x = x.view({-1, 512});
        auto logits = this->fc1->forward(x);
        x = torch::softmax(logits, /*dim=*/1);
        return x;
    }
};

#endif
