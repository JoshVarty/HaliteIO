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

        //TODO: Check if this works?
        if (!this->downsample.is_empty()) {
            identity = this->downsample->forward(x);
        }

        //TODO: Check if this works?
        out = out + identity;
        out = torch::relu(out);
        return out;
    }
};

struct Bottleneck : torch::nn::Module {
    static const int64_t EXPANSION = 4;

    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm bn1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm bn2;
    torch::nn::Conv2d conv3;
    torch::nn::BatchNorm bn3;
    torch::nn::Sequential downsample;

    Bottleneck(int64_t inplanes, int64_t planes, torch::nn::Sequential downsample)
    :   conv1(conv1x1(inplanes, planes, /*stride=*/1)),
        bn1(torch::nn::BatchNorm(torch::nn::BatchNormOptions(planes))),
        conv2(conv3x3(planes, planes, /*stride=*/1)),
        bn2(torch::nn::BatchNorm(torch::nn::BatchNormOptions(planes))),
        conv3(conv1x1(planes, planes * 4, /*stride=*/1)),
        bn3(torch::nn::BatchNorm(torch::nn::BatchNormOptions(planes * 4))),
        downsample(downsample)
    {
    }

    torch::Tensor forward(torch::Tensor x) {
        auto identity = x;
        
        auto out = conv1->forward(x);
        out = bn1->forward(out);
        out = torch::relu(out);

        out = conv2->forward(out);
        out = bn2->forward(out);
        out = torch::relu(out);

        out = conv3->forward(out);
        out = bn3->forward(out);

        //TODO: Check if this works?
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

    ResNet() {

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

    torch::nn::Sequential make_layer_bottleneck( int64_t planes, int64_t blocks, int64_t stride) {
        torch::nn::Sequential downsample;
        if(stride != 1 || this->inplanes != planes * Bottleneck::EXPANSION) {
            downsample = torch::nn::Sequential(
                conv1x1(this->inplanes, planes * Bottleneck::EXPANSION, stride),
                torch::nn::BatchNorm(planes * Bottleneck::EXPANSION)
            );
        }

        torch::nn::Sequential layers;
        auto newBlock = Bottleneck(this->inplanes, planes, downsample);
        layers->push_back(newBlock);
        this->inplanes = planes * Bottleneck::EXPANSION;

        for(int64_t i = 0; i < blocks; i++) {
            torch::nn::Sequential empty_downsample;
            newBlock = Bottleneck(this->inplanes, planes, downsample);
            layers->push_back(newBlock);
        }

        return layers;
    }
};


#endif
