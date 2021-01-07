//
// Created by 1337 on 1/8/2021.
//

#include <torch/torch.h>

#include "alexnet.h"

// Define namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;

AlexNetImpl::AlexNetImpl(int in_channels, int classes) {

    this->features = nn::Sequential(
            // layer 1 {C,227,277} -> {96,55,55}
            nn::Conv2d(
                    nn::Conv2dOptions(
                            /*in_channels=*/in_channels,
                            /*out_channels=*/96,
                            /*kernel_size=*/11
                    ).stride(4)
            ),
            nn::ReLU(nn::ReLUOptions().inplace(true)),
            nn::LocalResponseNorm(
                    nn::LocalResponseNormOptions(/*size=*/5).alpha(0.0001).beta(0.75).k(2.0)
            ),
            nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2)), // {96,55,55} -> {96,27,27}

            // layer 2 {96,27,27} ===> {256,27,27}
            nn::Conv2d(
                    nn::Conv2dOptions(
                            /*in_channels=*/96,
                            /*out_channels=*/256,
                            /*kernel_size=*/5
                    ).stride(1).padding(2)
            ),
            nn::ReLU(nn::ReLUOptions().inplace(true)),
            nn::LocalResponseNorm(nn::LocalResponseNormOptions(/*size=*/5).alpha(0.0001).beta(0.75).k(2.0)),
            nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2)), // {256,27,27} ===> {256,13,13}

            // layer 3 {256,13,13} ===> {384,13,13}
            nn::Conv2d(
                    nn::Conv2dOptions(
                            /*in_channels=*/256,
                            /*out_channels=*/384,
                            /*kernel_size=*/3
                    ).stride(1).padding(1)
            ),
            nn::ReLU(nn::ReLUOptions().inplace(true)),

            // layer 4 {384,13,13} ===> {384,13,13}
            nn::Conv2d(
                    nn::Conv2dOptions(
                            /*in_channels=*/384,
                            /*out_channels=*/384,
                            /*kernel_size=*/3
                    ).stride(1).padding(1)
            ),
            nn::ReLU(nn::ReLUOptions().inplace(true)),

            // layer 5 {384,13,13} ===> {256,13,13}
            nn::Conv2d(
                    nn::Conv2dOptions(
                            /*in_channels=*/384,
                            /*out_channels=*/256,
                            /*kernel_size=*/3
                    ).stride(1).padding(1)
            ),
            nn::ReLU(nn::ReLUOptions().inplace(true)),
            nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2)) // {256,13,13} ===> {256,6,6}
    );
    register_module("features", this->features);

    this->avgpool = nn::Sequential(
            nn::AdaptiveAvgPool2d(nn::AdaptiveAvgPool2dOptions({6, 6}))
    ); // {256,X,X} ===> {256,6,6}
    register_module("avgpool", this->avgpool);

    this->classifier = nn::Sequential(
            nn::Dropout(0.5),
            nn::Linear(/*in_channels=*/255*6*6, /*out_channels=*/4096), // {256*6*6} -> {4096}
            nn::ReLU(nn::ReLUOptions().inplace(true)),
            nn::Dropout(0.5),
            nn::Linear(/*in_channels=*/4096, /*out_channels=*/4096), // {4096} -> {4096}
            nn::ReLU(nn::ReLUOptions().inplace(true)),
            nn::Linear(/*in_channels=*/4096, /*out_channels=*/classes) // {4096} -> {number of classes}
    );
    register_module("classifier", this->classifier);

}

void AlexNetImpl::init() {
    this->apply(weights_init);
    nn::init::constant_(*(this->features[4]->named_parameters(false).find("bias")), /*bias=*/1.0);
    nn::init::constant_(*(this->features[10]->named_parameters(false).find("bias")), /*bias=*/1.0);
    nn::init::constant_(*(this->features[12]->named_parameters(false).find("bias")), /*bias=*/1.0);
}

torch::Tensor AlexNetImpl::forward(torch::Tensor x) {
    torch::Tensor feature, out;

    feature = this->features->forward(x);      // {C,227,227} ===> {256,6,6}
    feature = this->avgpool->forward(feature); // {256,X,X} ===> {256,6,6}
    feature = feature.view({feature.size(0), -1}); // flatten the tensor

    out = this->classifier->forward(feature);
    out = F::log_softmax(out, /*dim=*/1); // softmax activation

    return out;
}

/*
 * Print Network Layers
 * https://discuss.pytorch.org/t/print-network-architecture-in-cpp-jit/60297/2
 */
void AlexNetImpl::print_modules(size_t level) {
//    std::cout << this->features->name()qualifiedName() << " (\n";
//    for (const auto& module : module.get_modules()) {
//        print_tabs(level + 1);
//        this->print_modules(module.module, level + 1);
//    }
// TODO:
}

void weights_init(nn::Module &m) {
    if ((typeid(m) == typeid(nn::Conv2d)) || (typeid(m) == typeid(nn::Conv2dImpl))) {
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
    else if ((typeid(m) == typeid(nn::Linear)) || (typeid(m) == typeid(nn::LinearImpl))) {
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
        if (b != nullptr) nn::init::constant_(*b, /*bias=*/0.0);
    }
}

void print_tabs(size_t num) {
    for (size_t i = 0; i < num; i++) {
        std::cout << "\t";
    }
}
