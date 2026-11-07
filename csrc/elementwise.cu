// Memory-bound helpers: fused SiLU-and-mul for the gated MLP, and the scatter of new K/V into the paged cache.
#include "common.cuh"

namespace {

// x: [T, 2*I] laid out as [gate | up]; out: [T, I] = silu(gate) * up
__global__ void silu_and_mul_kernel(bf16* __restrict__ out, const bf16* __restrict__ x, int inter, int64_t x_stride,
                                    int64_t out_stride) {
  const int64_t row = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;  // vector index
  if (i * 8 >= inter) return;
  const bf16x8* gate = reinterpret_cast<const bf16x8*>(x + row * x_stride);
  const bf16x8* up = reinterpret_cast<const bf16x8*>(x + row * x_stride + inter);
  float g[8], u[8];
  unpack8(gate[i], g);
  unpack8(up[i], u);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const float s = round_bf16(g[k] / (1.f + __expf(-g[k])));
    g[k] = s * u[k];
  }
  reinterpret_cast<bf16x8*>(out + row * out_stride)[i] = pack8(g);
}

// Cache layout: [num_blocks, block_size, num_kv_heads, head_dim], i.e. one contiguous "slot" per token.
// slot_mapping[t] = physical_block * block_size + offset, or -1 for padding tokens (CUDA-graph batches).
__global__ void reshape_and_cache_kernel(const bf16* __restrict__ k, const bf16* __restrict__ v,
                                         bf16* __restrict__ k_cache, bf16* __restrict__ v_cache,
                                         const int64_t* __restrict__ slot_mapping, int64_t k_stride,
                                         int64_t v_stride, int heads, int head_dim) {
  const int64_t t = blockIdx.x;
  const int64_t slot = slot_mapping[t];
  if (slot < 0) return;
  const int per_head = head_dim / 8;
  const int nvec = heads * per_head;
  const int64_t base = slot * heads * head_dim;
  for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
    const int h = i / per_head;
    const int c = (i % per_head) * 8;
    const bf16x8 kv = *reinterpret_cast<const bf16x8*>(k + t * k_stride + h * head_dim + c);
    const bf16x8 vv = *reinterpret_cast<const bf16x8*>(v + t * v_stride + h * head_dim + c);
    *reinterpret_cast<bf16x8*>(k_cache + base + h * head_dim + c) = kv;
    *reinterpret_cast<bf16x8*>(v_cache + base + h * head_dim + c) = vv;
  }
}

}  // namespace

void silu_and_mul(torch::Tensor& out, const torch::Tensor& x) {
  CHECK_CUDA(x);
  CHECK_BF16(x);
  CHECK_BF16(out);
  CHECK_LAST_DIM_CONTIGUOUS(x);
  CHECK_LAST_DIM_CONTIGUOUS(out);
  const int T = x.size(0);
  const int inter = x.size(1) / 2;
  TORCH_CHECK(inter % 8 == 0 && out.size(1) == inter && out.size(0) == T);
  TORCH_CHECK(x.stride(0) % 8 == 0 && out.stride(0) % 8 == 0 && is_aligned16(x.data_ptr()) &&
              is_aligned16(out.data_ptr()));
  if (T == 0) return;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  constexpr int kThreads = 256;
  dim3 grid((inter / 8 + kThreads - 1) / kThreads, T);
  silu_and_mul_kernel<<<grid, kThreads, 0, current_stream()>>>(reinterpret_cast<bf16*>(out.data_ptr()),
                                                               reinterpret_cast<const bf16*>(x.data_ptr()), inter,
                                                               x.stride(0), out.stride(0));
}

// k, v: [T, Hk, D] views (token stride may be padded, head stride must be D).
void reshape_and_cache(const torch::Tensor& k, const torch::Tensor& v, torch::Tensor& k_cache, torch::Tensor& v_cache,
                       const torch::Tensor& slot_mapping) {
  CHECK_CUDA(k);
  CHECK_BF16(k);
  CHECK_BF16(v);
  CHECK_CONTIGUOUS(k_cache);
  CHECK_CONTIGUOUS(v_cache);
  CHECK_DTYPE(slot_mapping, at::kLong);
  TORCH_CHECK(k.dim() == 3 && v.dim() == 3);
  const int T = k.size(0);
  const int heads = k.size(1);
  const int head_dim = k.size(2);
  TORCH_CHECK(k.stride(2) == 1 && k.stride(1) == head_dim && v.stride(2) == 1 && v.stride(1) == head_dim);
  TORCH_CHECK(head_dim % 8 == 0 && k.stride(0) % 8 == 0 && v.stride(0) % 8 == 0);
  TORCH_CHECK(k_cache.size(2) == heads && k_cache.size(3) == head_dim);
  TORCH_CHECK(slot_mapping.numel() >= T);
  if (T == 0) return;
  const at::cuda::OptionalCUDAGuard guard(k.device());
  reshape_and_cache_kernel<<<T, 128, 0, current_stream()>>>(
      reinterpret_cast<const bf16*>(k.data_ptr()), reinterpret_cast<const bf16*>(v.data_ptr()),
      reinterpret_cast<bf16*>(k_cache.data_ptr()), reinterpret_cast<bf16*>(v_cache.data_ptr()),
      slot_mapping.data_ptr<int64_t>(), k.stride(0), v.stride(0), heads, head_dim);
}
