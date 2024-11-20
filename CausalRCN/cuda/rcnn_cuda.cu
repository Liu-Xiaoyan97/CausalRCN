#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void recursive_conv1d_kernel(float *x, const float *kernel, int b, int s, int d, int k) {
    // 获取线程 ID
    int batch_idx = blockIdx.x;  // 每个 block 处理一个 batch
    int feature_idx = threadIdx.x; // 每个线程处理一个特征维度
    int seq_len = s;

    // 开始递归卷积计算
    for (int i = 0; i < seq_len; i++){
        float y_i = 0.0;
        for (int j = 0; j < k; j ++){
            y_i += x[batch_idx * s * d + feature_idx * s + j];
        }
        x[batch_idx * s * d + feature_idx * s + i] = y_i;
        __syncthreads();
    }
}

// PyTorch 调用接口
void recursive_conv1d(torch::Tensor x, torch::Tensor kernel) {
    int b = x.size(0);
    int s = x.size(1);
    int d = x.size(2);
    int k = kernel.size(0);

    // 调用 CUDA kernel
    dim3 blocks(b);   // 每个 block 处理一个 batch
    dim3 threads(d);  // 每个线程处理输入特征维度上的一个元素

    // 启动 CUDA kernel
    recursive_conv1d_kernel<<<blocks, threads>>>(x.data_ptr<float>(), kernel.data_ptr<float>(), b, s, d, k);

    // 同步设备，确保计算完成
    cudaDeviceSynchronize();
}

// 前向卷积的反向传播核函数
__global__ void recursive_conv1d_backward_kernel(
    const float* grad_output, const float* x, const float* kernel,
    float* grad_x, float* grad_kernel, int b, int s, int d, int k) {

    // 获取线程 ID
    int batch_idx = blockIdx.x;  // 每个 block 处理一个 batch
    int feature_idx = threadIdx.x; // 每个线程处理一个特征维度
    int seq_len = s;

    // 每个 batch 的输入序列
    const float *input_seq = &x[batch_idx * s * d];  // 当前 batch 的输入序列
    const float *grad_y = &grad_output[batch_idx * s * d];  // 当前 batch 的输出梯度

    float *grad_input_seq = &grad_x[batch_idx * s * d];  // 当前 batch 的输入梯度
    float *grad_w = grad_kernel;  // 全局的卷积核梯度（所有 batch 共享）

    
    // 初始化梯度
    for (int i = 0; i < k * d; ++i) {
        grad_w[i] = 0.0f;
    }

    for (int i = seq_len -1; i >= 0; i--){
        float y_i = 0.0;
        for (int j = 0; j < k; j ++){
            &atomicAdd(grad_x[batch_idx * s * d + feature_idx * s + i], grad_w[feature_ids * s + j] * kernel[feature_ids * s + j]);
            &atomicAdd(grad_w[], g)
        }
        x[batch_idx * s * d + feature_idx * s + i] = y_i;
        __syncthreads();
    }

    // 从后向前遍历序列，计算梯度
    for (int i = seq_len - 1; i >= 0; --i) {
        if (feature_idx < d) {
            // 计算 grad_input_seq[i, feature_idx]
            for (int ki = 0; ki < k; ++ki) {
                if (i + ki < seq_len) {
                    // 累加梯度到输入
                    grad_input_seq[i * d + feature_idx] += grad_y[(i + ki) * d + feature_idx] * kernel[ki * d + feature_idx];
                }
            }
        }
        __syncthreads();

        // 计算卷积核的梯度
        if (feature_idx < d) {
            for (int ki = 0; ki < k; ++ki) {
                if (i + ki < seq_len) {
                    grad_w[ki * d + feature_idx] += grad_y[(i + ki) * d + feature_idx] * input_seq[i * d + feature_idx];
                }
            }
        }
        __syncthreads();
    }
}

// PyTorch 调用接口
void recursive_conv1d_backward(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor kernel,
    torch::Tensor grad_x, torch::Tensor grad_kernel) {

    int b = x.size(0);
    int s = x.size(1);
    int d = x.size(2);
    int k = kernel.size(0);

    // 调用 CUDA kernel
    dim3 blocks(b);   // 每个 block 处理一个 batch
    dim3 threads(d);  // 每个线程处理输入特征维度上的一个元素

    // 启动 CUDA kernel
    recursive_conv1d_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(), x.data_ptr<float>(), kernel.data_ptr<float>(),
        grad_x.data_ptr<float>(), grad_kernel.data_ptr<float>(), b, s, d, k);

    // 同步设备，确保计算完成
    cudaDeviceSynchronize();
}