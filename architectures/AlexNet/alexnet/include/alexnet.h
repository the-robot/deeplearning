#pragma once

#ifndef ALEXNET_ALEXNET_H

#define ALEXNET_ALEXNET_H
#include <torch/torch.h>

// Define namespace
namespace nn = torch::nn;

// Function prototype
void weights_init(nn::Module &m);
void print_tabs(size_t num);

struct AlexNetImpl : nn::Module {
private:
    nn::Sequential features, avgpool, classifier;

public:
    AlexNetImpl(){}
    AlexNetImpl(int in_channels, int classes);
    void init();
    torch::Tensor forward(torch::Tensor x);
    void print_modules(size_t level = 0);
};

TORCH_MODULE(AlexNet);

#endif //ALEXNET_CLEANEST_H
