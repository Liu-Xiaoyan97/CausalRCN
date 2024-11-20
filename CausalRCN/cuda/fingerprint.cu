#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>
#include <math.h>


__global__ void causal_cnn_forward(float* __restrict__ x, const float* __restrict__ y, int i, int b, int n, int d, int k){
    // int idx = threadIdx.x + threadIdx.y*blockDim.x + blockDim.x*blockDim.y*(blockIdx.x + blockIdx.y*gridDim.x);
    int idx_a = blockIdx.x * n * d + threadIdx.x * d;
    int idx_b = blockIdx.x * n * k + threadIdx.x * k;
    float tmp = 0.0f;
    assert(idx_a < b * n * d || idx_b < b * n * k);
    for (int j = 0; j < k; j ++){
        if (i - j < 0){
            break;
        }
        tmp += x[idx_a + i - j] * y[idx_b + k - 1 - j];
    }
    x[idx_a + i] = tmp;
}


__global__ void causal_cnn_backward(const  float* __restrict__ grad_output, 
                                    float* __restrict__ grad_x, 
                                    float* __restrict__ grad_y, 
                                    const float* __restrict__ x, 
                                    const float* __restrict__ y,
                                    const float* __restrict__ resdual, 
                                    int i, int b, int n, int d, int k) {
    int idx_a = blockIdx.x * n * d + threadIdx.x * d;
    int idx_b = blockIdx.x * n * k + threadIdx.x * k;
    assert(idx_a < b * n * d || idx_b < b * n * k);
    float grad_tmp = grad_output[idx_a + i];
    for (int j = 0; j < k; j++) {
        if (i - j < 0) {
            break;
        }
        if (j == k - 1){
            atomicAdd(&grad_y[idx_b + k - 1 - j], grad_tmp * resdual[idx_a + i]);
        }
        else{
            atomicAdd(&grad_y[idx_b + k - 1 - j], grad_tmp * x[idx_a + i]);
            atomicAdd(&grad_x[idx_a + i - j], grad_tmp * y[idx_b + k - 1 - j]);
        }
    }
}

torch::Tensor forward_cuda(torch::Tensor x, torch::Tensor y) {
    int b = x.size(0);
    int n = x.size(1);
    int d = x.size(2);
    int k = y.size(2);
    dim3 threadsPerBlock(n);
    dim3 numBlocks(b);
    for (int i = 0; i < d; i ++){
        causal_cnn_forward<<<numBlocks, threadsPerBlock>>>(x.data_ptr<float>(),
                                                        y.data_ptr<float>(), 
                                                        i, b, n, d, k);
        cudaDeviceSynchronize();
    }        
    return x;
}


std::vector<torch::Tensor> backward_cuda(torch::Tensor grad_output, torch::Tensor x, torch::Tensor y, torch::Tensor resdual) {
    int b = x.size(0);
    int n = x.size(1);
    int d = x.size(2);
    int k = y.size(2);

    auto grad_x = torch::zeros_like(x);
    auto grad_y = torch::zeros_like(y);

    dim3 threadsPerBlock(n);
    dim3 numBlocks(b);
    
    for (int i = d - 1; i >= 0; i --){
        causal_cnn_backward<<<numBlocks, threadsPerBlock>>>(grad_output.data_ptr<float>(), 
                                                            grad_x.data_ptr<float>(), 
                                                            grad_y.data_ptr<float>(),
                                                            x.data_ptr<float>(), 
                                                            y.data_ptr<float>(), 
                                                            resdual.data_ptr<float>(),
                                                            i, b, n, d, k);
        cudaDeviceSynchronize();
    }
    return {grad_x, grad_y};
}
