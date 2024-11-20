#include <torch/extension.h>

// Forward declaration of the forward and backward CUDA kernels
torch::Tensor forward_cuda(torch::Tensor x, torch::Tensor y);
std::vector<torch::Tensor> backward_cuda(torch::Tensor grad_output, torch::Tensor x, torch::Tensor y, torch::Tensor resdual);

// Forward and backward functions exposed to Python
torch::Tensor forward(torch::Tensor x, torch::Tensor y) {
    return forward_cuda(x, y);
}

std::vector<torch::Tensor> backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor y, torch::Tensor resdual) {
    return backward_cuda(grad_output, x, y, resdual);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom forward pass (CUDA)");
    m.def("backward", &backward, "Custom backward pass (CUDA)");
}
