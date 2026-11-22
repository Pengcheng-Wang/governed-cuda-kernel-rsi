#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {

__device__ __forceinline__ float silu(float value) {
  return value / (1.0f + __expf(-value));
}

__global__ void swiglu_bf16x2_kernel(
    const __nv_bfloat162* a,
    const __nv_bfloat162* b,
    __nv_bfloat162* output,
    int64_t pairs) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < pairs;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float2 a_values = __bfloat1622float2(a[index]);
    const float2 b_values = __bfloat1622float2(b[index]);
    output[index] = __floats2bfloat162_rn(
        silu(a_values.x) * b_values.x,
        silu(a_values.y) * b_values.y);
  }
}

__global__ void swiglu_bf16_tail_kernel(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* output,
    int64_t index) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const float a_value = __bfloat162float(a[index]);
    const float b_value = __bfloat162float(b[index]);
    output[index] = __float2bfloat16_rn(silu(a_value) * b_value);
  }
}

}  // namespace

torch::Tensor swiglu_cuda(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "a must be bfloat16");
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16, "b must be bfloat16");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(a.sizes() == b.sizes(), "input shapes must match");

  c10::cuda::CUDAGuard device_guard(a.device());
  auto output = torch::empty_like(a);
  const int64_t elements = a.numel();
  const int64_t pairs = elements / 2;
  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>(
      (pairs + threads - 1) / threads,
      4096));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();

  if (pairs > 0) {
    swiglu_bf16x2_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat162*>(b.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat162*>(output.data_ptr<at::BFloat16>()),
        pairs);
  }
  if (elements % 2 != 0) {
    swiglu_bf16_tail_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        elements - 1);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
