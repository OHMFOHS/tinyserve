#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

using bf16 = __nv_bfloat16;
using bf16x2 = __nv_bfloat162;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK((x).scalar_type() == at::kBFloat16, #x " must be bfloat16")
#define CHECK_DTYPE(x, t) TORCH_CHECK((x).scalar_type() == (t), #x " has wrong dtype")
#define CHECK_LAST_DIM_CONTIGUOUS(x) TORCH_CHECK((x).stride(-1) == 1, #x " last dim must be contiguous")

static inline cudaStream_t current_stream() { return at::cuda::getCurrentCUDAStream(); }

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Sum across the whole block. blockDim.x must be a multiple of 32. Result is broadcast to all threads.
__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float smem[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  const int num_warps = blockDim.x >> 5;
  v = (lane < num_warps) ? smem[lane] : 0.f;
  v = warp_reduce_sum(v);
  __syncthreads();  // smem may be reused by a following call
  return v;
}

// Round-trip through bf16 so fused kernels reproduce the rounding points of the eager PyTorch model.
__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16(x)); }

// 8 x bf16 packed in 16 bytes, used for 128-bit vectorized global memory access.
struct __align__(16) bf16x8 {
  bf16x2 v[4];
};

__device__ __forceinline__ void unpack8(const bf16x8& p, float* out) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 f = __bfloat1622float2(p.v[i]);
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

__device__ __forceinline__ bf16x8 pack8(const float* in) {
  bf16x8 p;
#pragma unroll
  for (int i = 0; i < 4; ++i) p.v[i] = __floats2bfloat162_rn(in[2 * i], in[2 * i + 1]);
  return p;
}

static inline bool is_aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }
