#include <torch/extension.h>

torch::Tensor swiglu_cuda(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("swiglu", &swiglu_cuda, "Vectorized BF16 SwiGLU (CUDA)");
}

