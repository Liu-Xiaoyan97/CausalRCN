#include <torch/extension.h>

// Forward declaration of the forward and backward CUDA kernels
torch::Tensor recursive_conv1d(torch::Tensor x, torch::Tensor kernel);
std::vector<torch::Tensor> recursive_conv1d_backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor kernel, torch::Tensor grad_x, torch::Tensor grad_kernel);

// Forward and backward functions exposed to Python
torch::Tensor forward(torch::Tensor x, torch::Tensor kernel) {
    return recursive_conv1d(x, kernel);
}

std::vector<torch::Tensor> backward(torch::Tensor grad_output, torch::Tensor x, torch::Tensor kernel, torch::Tensor grad_x, torch::Tensor grad_kernel) {
    return recursive_conv1d_backward(grad_output, x, kernel, grad_x, grad_kernel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom forward pass (CUDA)");
    m.def("backward", &backward, "Custom backward pass (CUDA)");
}
