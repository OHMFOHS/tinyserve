// Fused per-head RMSNorm (Qwen3 q_norm / k_norm) + NeoX-style rotary embedding, applied in place on the
// packed qkv projection output. One warp per (token, head): each lane owns D/32 consecutive elements, and the
// rotate-half partner of element j (j + D/2) lives exactly 16 lanes away, so the rotation is a single shuffle.
#include "common.cuh"

namespace {

template <int D>
__global__ void qk_norm_rope_kernel(bf16* __restrict__ qkv, int64_t token_stride, const int64_t* __restrict__ positions,
                                    const float* __restrict__ cos_sin,  // [max_pos, D]: cos (D/2) | sin (D/2)
                                    const bf16* __restrict__ q_w, const bf16* __restrict__ k_w, int num_tokens,
                                    int num_q_heads, int num_k_heads, float eps, bool do_norm) {
  static_assert(D % 64 == 0, "head_dim must be a multiple of 64");
  constexpr int E = D / 32;  // elements per lane
  const int lane = threadIdx.x & 31;
  const int64_t gw = static_cast<int64_t>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int heads = num_q_heads + num_k_heads;
  if (gw >= static_cast<int64_t>(num_tokens) * heads) return;  // whole warp exits together
  const int t = gw / heads;
  const int h = gw % heads;

  bf16* ptr = qkv + t * token_stride + h * D + lane * E;
  float x[E];
#pragma unroll
  for (int i = 0; i < E; i += 2) {
    float2 f = __bfloat1622float2(*reinterpret_cast<const bf16x2*>(ptr + i));
    x[i] = f.x;
    x[i + 1] = f.y;
  }

  if (do_norm) {
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < E; ++i) ss += x[i] * x[i];
    ss = warp_reduce_sum(ss);
    const float inv = rsqrtf(ss / D + eps);
    const bf16* w = (h < num_q_heads ? q_w : k_w) + lane * E;
#pragma unroll
    for (int i = 0; i < E; ++i) x[i] = round_bf16(__bfloat162float(w[i]) * round_bf16(x[i] * inv));
  }

  const float* cs = cos_sin + positions[t] * D;
  const bool first_half = lane < 16;
  float y[E];
#pragma unroll
  for (int i = 0; i < E; ++i) {
    const float partner = __shfl_xor_sync(0xffffffffu, x[i], 16);
    const int j = (lane * E + i) & (D / 2 - 1);
    const float c = cs[j];
    const float s = cs[D / 2 + j];
    y[i] = first_half ? (x[i] * c - partner * s) : (x[i] * c + partner * s);
  }
#pragma unroll
  for (int i = 0; i < E; i += 2) *reinterpret_cast<bf16x2*>(ptr + i) = __floats2bfloat162_rn(y[i], y[i + 1]);
}

}  // namespace

// qkv: [T, (Hq + 2*Hk) * D] (row stride may be padded). Only the q and k sections are modified.
void qk_norm_rope(torch::Tensor& qkv, const torch::Tensor& positions, const torch::Tensor& cos_sin,
                  const std::optional<torch::Tensor>& q_w, const std::optional<torch::Tensor>& k_w,
                  int64_t num_q_heads, int64_t num_k_heads, int64_t head_dim, double eps) {
  CHECK_CUDA(qkv);
  CHECK_BF16(qkv);
  CHECK_LAST_DIM_CONTIGUOUS(qkv);
  CHECK_DTYPE(positions, at::kLong);
  CHECK_DTYPE(cos_sin, at::kFloat);
  CHECK_CONTIGUOUS(cos_sin);
  TORCH_CHECK(cos_sin.size(1) == head_dim);
  const bool do_norm = q_w.has_value();
  TORCH_CHECK(do_norm == k_w.has_value(), "q_norm and k_norm weights must both be given or both omitted");
  const int T = qkv.size(0);
  if (T == 0) return;
  const at::cuda::OptionalCUDAGuard guard(qkv.device());

  const int64_t warps = static_cast<int64_t>(T) * (num_q_heads + num_k_heads);
  constexpr int kThreads = 256;
  const int blocks = (warps * 32 + kThreads - 1) / kThreads;
  const bf16* qw = do_norm ? reinterpret_cast<const bf16*>(q_w->data_ptr()) : nullptr;
  const bf16* kw = do_norm ? reinterpret_cast<const bf16*>(k_w->data_ptr()) : nullptr;
  auto* p = reinterpret_cast<bf16*>(qkv.data_ptr());
#define LAUNCH(D)                                                                                                  \
  qk_norm_rope_kernel<D><<<blocks, kThreads, 0, current_stream()>>>(p, qkv.stride(0), positions.data_ptr<int64_t>(), \
                                                                    cos_sin.data_ptr<float>(), qw, kw, T,           \
                                                                    num_q_heads, num_k_heads, eps, do_norm)
  switch (head_dim) {
    case 64: LAUNCH(64); break;
    case 128: LAUNCH(128); break;
    case 256: LAUNCH(256); break;
    default: TORCH_CHECK(false, "unsupported head_dim ", head_dim);
  }
#undef LAUNCH
}
