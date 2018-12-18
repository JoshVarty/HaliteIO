#ifndef RESNET_H
#define RESNET_H

#include <torch/torch.h>
#include "types.hpp"

torch::nn::Conv2d conv3x3(int64_t inputChannels, int64_t outputChannels, int64_t stride) {
    auto options = torch::nn::Conv2dOptions(inputChannels, outputChannels, /*kernel_size=*/3);
    options = options.stride(stride);
    return torch::nn::Conv2d(options);
}

torch::nn::Conv2d conv1x1(int64_t inputChannels, int64_t outputChannels, int64_t stride) {
    auto options = torch::nn::Conv2dOptions(inputChannels, outputChannels, /*kernel_size=*/3);
    options = options.stride(stride);
    return torch::nn::Conv2d(options);
}

struct BasicBlock : torch::nn::Module {
    static const int64_t EXPANSION = 1;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm bn1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm bn2;
    torch::nn::Sequential downsample;

    BasicBlock(int64_t inplanes, int64_t planes, torch::nn::Sequential downsample)
    :   conv1(conv3x3(inplanes, planes, /*stride*/1)),
        bn1(torch::nn::BatchNorm(torch::nn::BatchNormOptions(planes))),
        conv2(conv3x3(planes, planes, /*stride*/1)),
        bn2(torch::nn::BatchNorm(torch::nn::BatchNormOptions(planes))),
        downsample(downsample)
    {
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor selected_action) {
        auto identity = x;

        auto out = conv1->forward(x);
        out = bn1->forward(out);
        out = torch::relu(out);
        out = conv2->forward(out);
        out = bn2->forward(out);

        if (!this->downsample.is_empty()) {
            identity = this->downsample->forward(x);
        }

        //TODO: Check if this works?
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
    //torch::adaptive_avg_pool2d avgPool;

    ResNet(int64_t inputDepth, int layers[]) 
    :
    conv1Options(torch::nn::Conv2dOptions(inputDepth, 64, /*kernel_size=*/7).stride(2).padding(3).with_bias(false)),
    conv1(conv1Options),
    bn1(torch::nn::BatchNorm(64)),
    layer1(make_layer_basic(64,  layers[0], /*stride=*/1)),
    layer2(make_layer_basic(128, layers[1], /*stride=*/2)),
    layer3(make_layer_basic(256, layers[2], /*stride=*/2)),
    layer4(make_layer_basic(512, layers[3], /*stride=*/2))
    {
        //Init weights
        for (auto module_shared : this->modules()) {
            auto module = module_shared.get();
            if ( dynamic_cast<torch::nn::Conv2d*>(module)) {
                //TODO: Use Kaiming normal not xavier
                torch::nn:init::xavier_normal_(module.weight);
            }
            else if ( dynamic_cast<torch::nn::BatchNorm*>(module)) {
                torch::nn::init::constant_(module.weight, 1);
                torch::nn::init::constant_(module.bias, 0);
            }
        }
    }

    torch::nn::Sequential make_layer_basic(int64_t planes, int64_t blocks, int64_t stride) {

        torch::nn::Sequential downsample;
        if(stride != 1 || this->inplanes != planes * BasicBlock::EXPANSION) { //TODO: block.expansion?
            downsample = torch::nn::Sequential(
                conv1x1(this->inplanes, planes * BasicBlock::EXPANSION, stride),
                torch::nn::BatchNorm(planes * BasicBlock::EXPANSION)
            );
        }

        torch::nn::Sequential layers;
        auto newBlock = BasicBlock(this->inplanes, planes, downsample);
        layers->push_back(newBlock);
        this->inplanes = planes * BasicBlock::EXPANSION;

        for(int64_t i = 0; i < blocks; i++) {
            torch::nn::Sequential empty_downsample;
            newBlock = BasicBlock(this->inplanes, planes, empty_downsample);
            //layers->push_back(newBlock);
        }

        return layers;
    }
};


#endif
