// RMSNorm kernels: one thread block per token row, 128-bit vectorized loads, fp32 accumulation.
//
// Rounding mirrors HF Qwen3RMSNorm: normalize in fp32, round to bf16, then multiply by the bf16 weight.
#include "common.cuh"

namespace {

__global__ void rms_norm_kernel(bf16* __restrict__ out, const bf16* __restrict__ x, const bf16* __restrict__ w,
                                int hidden, int64_t x_stride, int64_t out_stride, float eps) {
  const int64_t row = blockIdx.x;
  const int nvec = hidden / 8;
  const bf16x8* xv = reinterpret_cast<const bf16x8*>(x + row * x_stride);
  const bf16x8* wv = reinterpret_cast<const bf16x8*>(w);
  bf16x8* ov = reinterpret_cast<bf16x8*>(out + row * out_stride);

  float ss = 0.f;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    float f[8];
    unpack8(xv[i], f);
#pragma unroll
    for (int k = 0; k < 8; ++k) ss += f[k] * f[k];
  }
  ss = block_reduce_sum(ss);
  const float inv = rsqrtf(ss / hidden + eps);

  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    float f[8], g[8];
    unpack8(xv[i], f);
    unpack8(wv[i], g);
#pragma unroll
    for (int k = 0; k < 8; ++k) f[k] = g[k] * round_bf16(f[k] * inv);
    ov[i] = pack8(f);
  }
}

// residual <- x + residual ; x <- rmsnorm(residual) * w   (both in place)
__global__ void fused_add_rms_norm_kernel(bf16* __restrict__ x, bf16* __restrict__ residual,
                                          const bf16* __restrict__ w, int hidden, int64_t x_stride,
                                          int64_t r_stride, float eps) {
  const int64_t row = blockIdx.x;
  const int nvec = hidden / 8;
  bf16x8* xv = reinterpret_cast<bf16x8*>(x + row * x_stride);
  bf16x8* rv = reinterpret_cast<bf16x8*>(residual + row * r_stride);
  const bf16x8* wv = reinterpret_cast<const bf16x8*>(w);

  float ss = 0.f;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    float a[8], b[8];
    unpack8(xv[i], a);
    unpack8(rv[i], b);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      b[k] = round_bf16(a[k] + b[k]);
      ss += b[k] * b[k];
    }
    rv[i] = pack8(b);
  }
  ss = block_reduce_sum(ss);
  const float inv = rsqrtf(ss / hidden + eps);

  // Each thread re-reads only the residual vectors it wrote itself, so no extra barrier is needed.
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    float b[8], g[8];
    unpack8(rv[i], b);
    unpack8(wv[i], g);
#pragma unroll
    for (int k = 0; k < 8; ++k) b[k] = g[k] * round_bf16(b[k] * inv);
    xv[i] = pack8(b);
  }
}

int pick_threads(int hidden) {
  int nvec = hidden / 8;
  int t = ((nvec + 31) / 32) * 32;
  return t > 1024 ? 1024 : (t < 32 ? 32 : t);
}

void check_rows(const torch::Tensor& t, const char* name) {
  CHECK_CUDA(t);
  CHECK_BF16(t);
  TORCH_CHECK(t.dim() == 2, name, " must be 2D [tokens, hidden]");
  CHECK_LAST_DIM_CONTIGUOUS(t);
  TORCH_CHECK(t.stride(0) % 8 == 0 && is_aligned16(t.data_ptr()), name, " must be 16-byte aligned per row");
}

}  // namespace

void rms_norm(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& w, double eps) {
  check_rows(x, "x");
  check_rows(out, "out");
  CHECK_CONTIGUOUS(w);
  const int hidden = x.size(1);
  TORCH_CHECK(hidden % 8 == 0 && w.numel() == hidden && out.size(1) == hidden);
  const int rows = x.size(0);
  if (rows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  rms_norm_kernel<<<rows, pick_threads(hidden), 0, current_stream()>>>(
      reinterpret_cast<bf16*>(out.data_ptr()), reinterpret_cast<const bf16*>(x.data_ptr()),
      reinterpret_cast<const bf16*>(w.data_ptr()), hidden, x.stride(0), out.stride(0), static_cast<float>(eps));
}

void fused_add_rms_norm(torch::Tensor& x, torch::Tensor& residual, const torch::Tensor& w, double eps) {
  check_rows(x, "x");
  check_rows(residual, "residual");
  CHECK_CONTIGUOUS(w);
  const int hidden = x.size(1);
  TORCH_CHECK(hidden % 8 == 0 && w.numel() == hidden && residual.size(1) == hidden);
  TORCH_CHECK(residual.size(0) == x.size(0));
  const int rows = x.size(0);
  if (rows == 0) return;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  fused_add_rms_norm_kernel<<<rows, pick_threads(hidden), 0, current_stream()>>>(
      reinterpret_cast<bf16*>(x.data_ptr()), reinterpret_cast<bf16*>(residual.data_ptr()),
      reinterpret_cast<const bf16*>(w.data_ptr()), hidden, x.stride(0), residual.stride(0), static_cast<float>(eps));
}
