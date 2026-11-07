// Paged attention for the decode phase (one query token per sequence), flash-decoding style.
//
// Grid = (num_kv_heads, batch, num_partitions). Each thread block handles one KV head of one sequence for one
// slice of PART tokens, and serves all G query heads that share that KV head (GQA), so every K/V row is read
// from HBM once per group rather than once per query head. Inside a block, 8-lane groups each own one token at a
// time (8 lanes x 16 elements = one 128-dim head), so a dot product needs only a 3-step shuffle reduction. Each
// group keeps its own online-softmax state; groups, then warps, then partitions are merged with log-sum-exp
// rescaling. When a sequence fits in one partition the block writes the final output directly.
#include "common.cuh"

namespace {

constexpr int kPartSize = 256;
constexpr int kNumWarps = 4;
constexpr int kGroup = 8;  // lanes cooperating on one token

template <int D, int G>
__global__ void __launch_bounds__(kNumWarps * 32)
    paged_decode_kernel(bf16* __restrict__ out, int64_t out_stride, float* __restrict__ tmp_out,
                        float* __restrict__ tmp_ml, const bf16* __restrict__ q, int64_t q_stride,
                        const bf16* __restrict__ k_cache, const bf16* __restrict__ v_cache,
                        const int* __restrict__ block_tables, int max_blocks_per_seq,
                        const int* __restrict__ context_lens, int block_size, int num_kv_heads, float scale_log2,
                        int num_partitions) {
  constexpr int EPL = D / kGroup;  // elements per lane
  constexpr int NV = EPL / 8;      // 16-byte vectors per lane
  constexpr int WORKERS = kNumWarps * (32 / kGroup);
  constexpr int UNROLL = 2;

  const int hk = blockIdx.x;
  const int b = blockIdx.y;
  const int part = blockIdx.z;
  const int ctx = context_lens[b];
  const int start = part * kPartSize;
  if (start >= ctx) return;  // block-uniform
  const int end = min(ctx, start + kPartSize);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int gl = lane & (kGroup - 1);
  const int worker = warp * (32 / kGroup) + lane / kGroup;
  const int num_q_heads = num_kv_heads * G;

  float qr[G][EPL];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const bf16x8* qp = reinterpret_cast<const bf16x8*>(q + b * q_stride + (hk * G + g) * D + gl * EPL);
#pragma unroll
    for (int v = 0; v < NV; ++v) {
      unpack8(qp[v], &qr[g][v * 8]);
#pragma unroll
      for (int e = 0; e < 8; ++e) qr[g][v * 8 + e] *= scale_log2;
    }
  }

  float m[G], l[G], acc[G][EPL];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    m[g] = -INFINITY;
    l[g] = 0.f;
#pragma unroll
    for (int e = 0; e < EPL; ++e) acc[g][e] = 0.f;
  }

  const int* bt = block_tables + static_cast<int64_t>(b) * max_blocks_per_seq;
  const int64_t slot_stride = static_cast<int64_t>(num_kv_heads) * D;
  const int64_t head_off = static_cast<int64_t>(hk) * D + gl * EPL;

  for (int base = start; base < end; base += WORKERS * UNROLL) {
    bool valid[UNROLL];
    bf16x8 kr[UNROLL][NV], vr[UNROLL][NV];
    // Issue all loads for this iteration before any math, to keep several rows in flight per thread.
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int tok = base + worker + u * WORKERS;
      valid[u] = tok < end;
      if (valid[u]) {
        const int64_t slot = static_cast<int64_t>(bt[tok / block_size]) * block_size + tok % block_size;
        const bf16x8* kp = reinterpret_cast<const bf16x8*>(k_cache + slot * slot_stride + head_off);
        const bf16x8* vp = reinterpret_cast<const bf16x8*>(v_cache + slot * slot_stride + head_off);
#pragma unroll
        for (int v = 0; v < NV; ++v) {
          kr[u][v] = kp[v];
          vr[u][v] = vp[v];
        }
      } else {
#pragma unroll
        for (int v = 0; v < NV; ++v) {
          kr[u][v] = bf16x8{};
          vr[u][v] = bf16x8{};
        }
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      float kf[EPL], vf[EPL];
#pragma unroll
      for (int v = 0; v < NV; ++v) {
        unpack8(kr[u][v], &kf[v * 8]);
        unpack8(vr[u][v], &vf[v * 8]);
      }
#pragma unroll
      for (int g = 0; g < G; ++g) {
        float s = 0.f;
#pragma unroll
        for (int e = 0; e < EPL; ++e) s += qr[g][e] * kf[e];
        // Reduce inside the 8-lane group. All lanes execute this, valid or not, so the shuffle is convergent.
        s += __shfl_xor_sync(0xffffffffu, s, 4);
        s += __shfl_xor_sync(0xffffffffu, s, 2);
        s += __shfl_xor_sync(0xffffffffu, s, 1);
        if (valid[u]) {
          const float m_new = fmaxf(m[g], s);
          const float alpha = exp2f(m[g] - m_new);
          const float p = exp2f(s - m_new);
          l[g] = l[g] * alpha + p;
#pragma unroll
          for (int e = 0; e < EPL; ++e) acc[g][e] = acc[g][e] * alpha + p * vf[e];
          m[g] = m_new;
        }
      }
    }
  }

  // Merge the 4 groups of each warp (lanes l, l^8, l^16, l^24 hold the same dims).
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float M = m[g];
    M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 8));
    M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 16));
    const float sc = (m[g] == -INFINITY) ? 0.f : exp2f(m[g] - M);
    float L = l[g] * sc;
    L += __shfl_xor_sync(0xffffffffu, L, 8);
    L += __shfl_xor_sync(0xffffffffu, L, 16);
#pragma unroll
    for (int e = 0; e < EPL; ++e) {
      float a = acc[g][e] * sc;
      a += __shfl_xor_sync(0xffffffffu, a, 8);
      a += __shfl_xor_sync(0xffffffffu, a, 16);
      acc[g][e] = a;
    }
    m[g] = M;
    l[g] = L;
  }

  __shared__ float s_m[kNumWarps][G];
  __shared__ float s_l[kNumWarps][G];
  __shared__ float s_acc[kNumWarps][G][D];
  if (lane < kGroup) {
#pragma unroll
    for (int g = 0; g < G; ++g) {
#pragma unroll
      for (int e = 0; e < EPL; ++e) s_acc[warp][g][gl * EPL + e] = acc[g][e];
      if (lane == 0) {
        s_m[warp][g] = m[g];
        s_l[warp][g] = l[g];
      }
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < G * D; i += blockDim.x) {
    const int g = i / D;
    const int d = i % D;
    float M = -INFINITY;
#pragma unroll
    for (int w = 0; w < kNumWarps; ++w) M = fmaxf(M, s_m[w][g]);
    float L = 0.f, O = 0.f;
#pragma unroll
    for (int w = 0; w < kNumWarps; ++w) {
      const float sc = (s_m[w][g] == -INFINITY) ? 0.f : exp2f(s_m[w][g] - M);
      L += s_l[w][g] * sc;
      O += s_acc[w][g][d] * sc;
    }
    const int hq = hk * G + g;
    if (num_partitions == 1) {
      out[b * out_stride + hq * D + d] = __float2bfloat16(O / L);
    } else {
      const int64_t idx = (static_cast<int64_t>(b) * num_q_heads + hq) * num_partitions + part;
      tmp_out[idx * D + d] = O / L;
      if (d == 0) {
        tmp_ml[idx * 2] = M;
        tmp_ml[idx * 2 + 1] = L;
      }
    }
  }
}

// Combine per-partition partial results: out = sum_p L_p * 2^(M_p - M) * O_p / sum_p L_p * 2^(M_p - M)
template <int D>
__global__ void decode_reduce_kernel(bf16* __restrict__ out, int64_t out_stride, const float* __restrict__ tmp_out,
                                     const float* __restrict__ tmp_ml, const int* __restrict__ context_lens,
                                     int num_q_heads, int num_partitions) {
  const int hq = blockIdx.x;
  const int b = blockIdx.y;
  const int d = threadIdx.x;
  const int ctx = context_lens[b];
  const int np = (ctx + kPartSize - 1) / kPartSize;
  bf16* o = out + b * out_stride + hq * D + d;
  if (np == 0) {  // padding slot of a CUDA-graph batch
    *o = __float2bfloat16(0.f);
    return;
  }
  const int64_t base = (static_cast<int64_t>(b) * num_q_heads + hq) * num_partitions;
  float M = -INFINITY;
  for (int p = 0; p < np; ++p) M = fmaxf(M, tmp_ml[(base + p) * 2]);
  float W = 0.f, O = 0.f;
  for (int p = 0; p < np; ++p) {
    const float w = tmp_ml[(base + p) * 2 + 1] * exp2f(tmp_ml[(base + p) * 2] - M);
    W += w;
    O += w * tmp_out[(base + p) * D + d];
  }
  *o = __float2bfloat16(O / W);
}

}  // namespace

int64_t decode_partition_size() { return kPartSize; }

// q: [B, Hq, D] (token stride may be padded); out: [B, Hq, D]; caches: [num_blocks, block_size, Hk, D]
// num_partitions must be >= ceil(max(context_lens) / partition_size). With num_partitions > 1, tmp_out must be
// float [B, Hq, P, D] and tmp_ml float [B, Hq, P, 2]. Fixing num_partitions keeps the launch CUDA-graph safe.
void paged_decode_attention(torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& k_cache,
                            const torch::Tensor& v_cache, const torch::Tensor& block_tables,
                            const torch::Tensor& context_lens, double scale, int64_t num_partitions,
                            torch::Tensor& tmp_out, torch::Tensor& tmp_ml) {
  CHECK_CUDA(q);
  CHECK_BF16(q);
  CHECK_BF16(out);
  CHECK_BF16(k_cache);
  CHECK_CONTIGUOUS(k_cache);
  CHECK_CONTIGUOUS(v_cache);
  CHECK_DTYPE(block_tables, at::kInt);
  CHECK_DTYPE(context_lens, at::kInt);
  CHECK_CONTIGUOUS(block_tables);
  TORCH_CHECK(q.dim() == 3 && out.dim() == 3);
  const int B = q.size(0);
  const int Hq = q.size(1);
  const int D = q.size(2);
  const int block_size = k_cache.size(1);
  const int Hk = k_cache.size(2);
  TORCH_CHECK(k_cache.size(3) == D && Hq % Hk == 0);
  TORCH_CHECK(q.stride(2) == 1 && q.stride(1) == D && out.stride(2) == 1 && out.stride(1) == D);
  TORCH_CHECK(q.stride(0) % 8 == 0 && is_aligned16(q.data_ptr()));
  TORCH_CHECK(num_partitions >= 1);
  if (num_partitions > 1) {
    CHECK_DTYPE(tmp_out, at::kFloat);
    CHECK_DTYPE(tmp_ml, at::kFloat);
    TORCH_CHECK(tmp_out.numel() >= static_cast<int64_t>(B) * Hq * num_partitions * D);
    TORCH_CHECK(tmp_ml.numel() >= static_cast<int64_t>(B) * Hq * num_partitions * 2);
  }
  if (B == 0) return;
  const at::cuda::OptionalCUDAGuard guard(q.device());
  const int G = Hq / Hk;
  const float scale_log2 = static_cast<float>(scale) * 1.4426950408889634f;
  float* tmp_o = num_partitions > 1 ? tmp_out.data_ptr<float>() : nullptr;
  float* tmp_m = num_partitions > 1 ? tmp_ml.data_ptr<float>() : nullptr;
  auto* o = reinterpret_cast<bf16*>(out.data_ptr());
  dim3 grid(Hk, B, num_partitions);
  auto stream = current_stream();

#define LAUNCH(D_, G_)                                                                                            \
  paged_decode_kernel<D_, G_><<<grid, kNumWarps * 32, 0, stream>>>(                                               \
      o, out.stride(0), tmp_o, tmp_m, reinterpret_cast<const bf16*>(q.data_ptr()), q.stride(0),                   \
      reinterpret_cast<const bf16*>(k_cache.data_ptr()), reinterpret_cast<const bf16*>(v_cache.data_ptr()),       \
      block_tables.data_ptr<int>(), block_tables.size(1), context_lens.data_ptr<int>(), block_size, Hk, scale_log2, \
      num_partitions)
#define DISPATCH_G(D_)                                    \
  switch (G) {                                            \
    case 1: LAUNCH(D_, 1); break;                         \
    case 2: LAUNCH(D_, 2); break;                         \
    case 4: LAUNCH(D_, 4); break;                         \
    case 8: LAUNCH(D_, 8); break;                         \
    default: TORCH_CHECK(false, "unsupported GQA group size ", G); \
  }
  switch (D) {
    case 64: DISPATCH_G(64); break;
    case 128: DISPATCH_G(128); break;
    default: TORCH_CHECK(false, "unsupported head_dim ", D);
  }
#undef DISPATCH_G
#undef LAUNCH

  if (num_partitions > 1) {
    dim3 rgrid(Hq, B);
    if (D == 64) {
      decode_reduce_kernel<64><<<rgrid, 64, 0, stream>>>(o, out.stride(0), tmp_o, tmp_m, context_lens.data_ptr<int>(),
                                                         Hq, num_partitions);
    } else {
      decode_reduce_kernel<128><<<rgrid, 128, 0, stream>>>(o, out.stride(0), tmp_o, tmp_m,
                                                           context_lens.data_ptr<int>(), Hq, num_partitions);
    }
  }
}
