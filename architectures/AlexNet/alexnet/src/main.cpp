#include <iostream>
#include <torch/torch.h>

#include "alexnet.h" // AlexNet

// Function prototype
torch::Device GetGPU();

int main() {
    std::cout << "AlexNet Implementation" << std::endl;

    torch::Device device = GetGPU();

    // define network
    AlexNet model(/*in_channels=*/255, /*classes=*/10);
    model->to(device);

    model->print_modules();

    return 0;
}

/*
 * Get GPU Device
 */
torch::Device GetGPU() {
    if (torch::cuda::is_available()) {
        std::cout << "GPU Available" << std::endl;
    }

    torch::Device device(torch::kCUDA);
    return device;
}
