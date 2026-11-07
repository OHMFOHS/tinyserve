// Causal varlen prefill attention over the paged KV cache, FlashAttention-2 style, on tensor cores.
//
// The new tokens' K/V are written into the cache first (reshape_and_cache), then this kernel attends each query
// to every cached key up to its own position. Reading K/V through the block table means the same kernel serves
// plain prefill, prefix-cache hits (the cached prefix is never recomputed) and chunked prefill.
//
// Grid = (ceil(max_q_len / 64), num_q_heads, batch); 4 warps, each owning 16 query rows. Per 64-key tile:
//   S = Q K^T   with mma.sync.m16n8k16 (bf16 in, fp32 accumulate); Q fragments live in registers, K fragments
//               come from shared memory via ldmatrix
//   online softmax on the S accumulators (row max / sum via quad shuffles, exp2 with pre-scaled logits)
//   O += P V    the S accumulator layout is reused directly as the A operand (P never touches shared memory);
//               V fragments via ldmatrix.trans
// K and V tiles are fetched with cp.async into separate buffers, so loading K[j+1] overlaps softmax + P·V of
// tile j, and loading V[j+1] overlaps Q·K^T of tile j+1.
#include "common.cuh"

namespace {

constexpr int kBM = 64;   // query rows per block
constexpr int kBN = 64;   // keys per tile
constexpr int kPad = 8;   // smem row padding (bf16 elements): row stride 272 B keeps ldmatrix conflict-free
constexpr int kThreads = 128;

__device__ __forceinline__ uint32_t ld32(const bf16* p) { return *reinterpret_cast<const uint32_t*>(p); }

__device__ __forceinline__ uint32_t pack_f32(float lo, float hi) {
  bf16x2 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16-byte async global->shared copy; src_bytes == 0 zero-fills the destination without reading global memory.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(dst)), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const bf16* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(p)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* r, const bf16* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(p)));
}

// D(16x8, f32) += A(16x16, bf16, row-major) * B(16x8, bf16, col-major)
__device__ __forceinline__ void mma_16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Issue the cp.async copies of one 64-token K or V tile of a sequence's paged cache into shared memory.
template <int D>
__device__ __forceinline__ void load_tile_async(bf16* dst, const bf16* __restrict__ cache, const int* bt, int j0,
                                                int ctx, int block_size, int64_t slot_stride, int head_off) {
  constexpr int LDS = D + kPad;
  constexpr int VPR = D / 8;
#pragma unroll
  for (int i = threadIdx.x; i < kBN * VPR; i += kThreads) {
    const int r = i / VPR;
    const int c = (i % VPR) * 8;
    const int tok = j0 + r;
    const bf16* src = cache;  // any valid address when zero-filling
    int bytes = 0;
    if (tok < ctx) {
      const int64_t slot = static_cast<int64_t>(bt[tok / block_size]) * block_size + tok % block_size;
      src = cache + slot * slot_stride + head_off + c;
      bytes = 16;
    }
    cp_async16(dst + r * LDS + c, src, bytes);
  }
}

template <int D>
__global__ void __launch_bounds__(kThreads)
    paged_prefill_kernel(bf16* __restrict__ out, int64_t out_stride, const bf16* __restrict__ q, int64_t q_stride,
                         const bf16* __restrict__ k_cache, const bf16* __restrict__ v_cache,
                         const int* __restrict__ block_tables, int max_blocks_per_seq,
                         const int* __restrict__ cu_seqlens_q, const int* __restrict__ context_lens, int block_size,
                         int num_kv_heads, int group, float scale_log2) {
  constexpr int LDS = D + kPad;
  constexpr int KSTEPS = D / 16;  // k-steps of Q K^T
  constexpr int DT = D / 8;       // n-tiles of the output
  constexpr int NT = kBN / 8;     // n-tiles of S
  __shared__ __align__(128) bf16 Ks[kBN * LDS];
  __shared__ __align__(128) bf16 Vs[kBN * LDS];

  const int mt = blockIdx.x;
  const int hq = blockIdx.y;
  const int b = blockIdx.z;
  const int q_start = cu_seqlens_q[b];
  const int q_len = cu_seqlens_q[b + 1] - q_start;
  if (mt * kBM >= q_len) return;  // block-uniform
  const int ctx = context_lens[b];
  const int prefix = ctx - q_len;  // tokens already in the cache before this chunk
  const int hk = hq / group;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;  // row within the 8-row group of an mma fragment
  const int t4 = lane & 3;  // column pair index
  const int lm_row = lane & 7;   // ldmatrix: row this lane addresses
  const int lm_mat = lane >> 3;  // ldmatrix: which of the 4 matrices

  const int kv_end = prefix + min(q_len, (mt + 1) * kBM);  // exclusive: last key any row of this block needs
  const int* bt = block_tables + static_cast<int64_t>(b) * max_blocks_per_seq;
  const int64_t slot_stride = static_cast<int64_t>(num_kv_heads) * D;
  const int head_off = hk * D;

  // Start fetching the first K and V tiles before touching Q.
  load_tile_async<D>(Ks, k_cache, bt, 0, ctx, block_size, slot_stride, head_off);
  cp_async_commit();
  load_tile_async<D>(Vs, v_cache, bt, 0, ctx, block_size, slot_stride, head_off);
  cp_async_commit();

  const int row0 = mt * kBM + warp * 16;
  const int ra = row0 + g;
  const int rb = ra + 8;
  const bool va = ra < q_len;
  const bool vb = rb < q_len;

  uint32_t qf[KSTEPS][4];
  {
    const bf16* qa = q + static_cast<int64_t>(q_start + ra) * q_stride + hq * D + t4 * 2;
    const bf16* qb = q + static_cast<int64_t>(q_start + rb) * q_stride + hq * D + t4 * 2;
#pragma unroll
    for (int ks = 0; ks < KSTEPS; ++ks) {
      qf[ks][0] = va ? ld32(qa + ks * 16) : 0u;
      qf[ks][1] = vb ? ld32(qb + ks * 16) : 0u;
      qf[ks][2] = va ? ld32(qa + ks * 16 + 8) : 0u;
      qf[ks][3] = vb ? ld32(qb + ks * 16 + 8) : 0u;
    }
  }

  float o[DT][4];
#pragma unroll
  for (int i = 0; i < DT; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.f, l_b = 0.f;
  const int qpos_a = prefix + ra;
  const int qpos_b = prefix + rb;
  // Last key this warp needs; warps whose rows are all padding skip compute entirely.
  const int warp_last = row0 < q_len ? prefix + min(row0 + 15, q_len - 1) : -1;

  for (int j0 = 0; j0 < kv_end; j0 += kBN) {
    const bool has_next = j0 + kBN < kv_end;
    const bool active = j0 <= warp_last;  // warp-uniform; otherwise the tile is in this warp's causal future

    cp_async_wait<1>();  // K[j] landed (V[j] may still be in flight)
    __syncthreads();

    float s[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
    if (active) {
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int ks = 0; ks < KSTEPS; ks += 2) {
          // 4 matrices: keys nt*8..+8 x dims [ks*16, ks*16+32) in 8-wide slices -> b0/b1 of k-steps ks, ks+1
          uint32_t kb[4];
          ldmatrix_x4(kb, &Ks[(nt * 8 + lm_row) * LDS + ks * 16 + lm_mat * 8]);
          mma_16816(s[nt], qf[ks], kb[0], kb[1]);
          mma_16816(s[nt], qf[ks + 1], kb[2], kb[3]);
        }
      }
    }
    __syncthreads();  // everyone is done reading Ks
    if (has_next) {
      load_tile_async<D>(Ks, k_cache, bt, j0 + kBN, ctx, block_size, slot_stride, head_off);
      cp_async_commit();
    }

    if (active) {
      // Scale into the log2 domain and apply the causal mask (only on tiles that cross this warp's diagonal).
      const bool need_mask = j0 + kBN - 1 > prefix + row0;
      float mx_a = -INFINITY, mx_b = -INFINITY;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int key = j0 + nt * 8 + t4 * 2 + (c & 1);
          float v = s[nt][c] * scale_log2;
          if (need_mask && key > (c < 2 ? qpos_a : qpos_b)) v = -INFINITY;
          s[nt][c] = v;
        }
        mx_a = fmaxf(mx_a, fmaxf(s[nt][0], s[nt][1]));
        mx_b = fmaxf(mx_b, fmaxf(s[nt][2], s[nt][3]));
      }
      mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 1));
      mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 2));
      mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 1));
      mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 2));

      const float mn_a = fmaxf(m_a, mx_a);
      const float mn_b = fmaxf(m_b, mx_b);
      const float base_a = mn_a == -INFINITY ? 0.f : mn_a;  // fully-masked rows must not produce NaN
      const float base_b = mn_b == -INFINITY ? 0.f : mn_b;
      const float alpha_a = exp2f(m_a - base_a);
      const float alpha_b = exp2f(m_b - base_b);
      m_a = mn_a;
      m_b = mn_b;

      float sum_a = 0.f, sum_b = 0.f;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        s[nt][0] = exp2f(s[nt][0] - base_a);
        s[nt][1] = exp2f(s[nt][1] - base_a);
        s[nt][2] = exp2f(s[nt][2] - base_b);
        s[nt][3] = exp2f(s[nt][3] - base_b);
        sum_a += s[nt][0] + s[nt][1];
        sum_b += s[nt][2] + s[nt][3];
      }
      l_a = l_a * alpha_a + sum_a;  // lane-partial row sums; reduced across the quad at the end
      l_b = l_b * alpha_b + sum_b;
#pragma unroll
      for (int dt = 0; dt < DT; ++dt) {
        o[dt][0] *= alpha_a;
        o[dt][1] *= alpha_a;
        o[dt][2] *= alpha_b;
        o[dt][3] *= alpha_b;
      }
    }

    if (has_next) cp_async_wait<1>();  // V[j] landed; K[j+1] may still be in flight
    else cp_async_wait<0>();
    __syncthreads();

    if (active) {
      // O += P V. The C layout of two adjacent 16x8 S tiles is exactly the A layout of one 16x16 P tile.
#pragma unroll
      for (int kk = 0; kk < kBN / 16; ++kk) {
        uint32_t a[4];
        a[0] = pack_f32(s[2 * kk][0], s[2 * kk][1]);
        a[1] = pack_f32(s[2 * kk][2], s[2 * kk][3]);
        a[2] = pack_f32(s[2 * kk + 1][0], s[2 * kk + 1][1]);
        a[3] = pack_f32(s[2 * kk + 1][2], s[2 * kk + 1][3]);
        // 4 matrices: keys {kk*16, kk*16+8} x dims {dt*8, dt*8+8}, transposed -> b0/b1 of n-tiles dt, dt+1
        const bf16* vbase = &Vs[(kk * 16 + (lm_mat & 1) * 8 + lm_row) * LDS + (lm_mat >> 1) * 8];
#pragma unroll
        for (int dt = 0; dt < DT; dt += 2) {
          uint32_t vb4[4];
          ldmatrix_x4_trans(vb4, vbase + dt * 8);
          mma_16816(o[dt], a, vb4[0], vb4[1]);
          mma_16816(o[dt + 1], a, vb4[2], vb4[3]);
        }
      }
    }
    __syncthreads();  // everyone is done reading Vs
    if (has_next) {
      load_tile_async<D>(Vs, v_cache, bt, j0 + kBN, ctx, block_size, slot_stride, head_off);
      cp_async_commit();
    }
  }

  l_a += __shfl_xor_sync(0xffffffffu, l_a, 1);
  l_a += __shfl_xor_sync(0xffffffffu, l_a, 2);
  l_b += __shfl_xor_sync(0xffffffffu, l_b, 1);
  l_b += __shfl_xor_sync(0xffffffffu, l_b, 2);
  const float inv_a = l_a > 0.f ? 1.f / l_a : 0.f;
  const float inv_b = l_b > 0.f ? 1.f / l_b : 0.f;
  if (va) {
    bf16* oa = out + static_cast<int64_t>(q_start + ra) * out_stride + hq * D + t4 * 2;
#pragma unroll
    for (int dt = 0; dt < DT; ++dt)
      *reinterpret_cast<bf16x2*>(oa + dt * 8) = __floats2bfloat162_rn(o[dt][0] * inv_a, o[dt][1] * inv_a);
  }
  if (vb) {
    bf16* ob = out + static_cast<int64_t>(q_start + rb) * out_stride + hq * D + t4 * 2;
#pragma unroll
    for (int dt = 0; dt < DT; ++dt)
      *reinterpret_cast<bf16x2*>(ob + dt * 8) = __floats2bfloat162_rn(o[dt][2] * inv_b, o[dt][3] * inv_b);
  }
}

}  // namespace

// q: [Tq, Hq, D] packed varlen queries (token stride may be padded); out: [Tq, Hq, D]
// cu_seqlens_q: [B+1] int32 query offsets; context_lens: [B] int32 total kv length (cached prefix + new tokens)
void paged_prefill_attention(torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& k_cache,
                             const torch::Tensor& v_cache, const torch::Tensor& block_tables,
                             const torch::Tensor& cu_seqlens_q, const torch::Tensor& context_lens, int64_t max_q_len,
                             double scale) {
  CHECK_CUDA(q);
  CHECK_BF16(q);
  CHECK_BF16(out);
  CHECK_BF16(k_cache);
  CHECK_CONTIGUOUS(k_cache);
  CHECK_CONTIGUOUS(v_cache);
  CHECK_CONTIGUOUS(block_tables);
  CHECK_DTYPE(block_tables, at::kInt);
  CHECK_DTYPE(cu_seqlens_q, at::kInt);
  CHECK_DTYPE(context_lens, at::kInt);
  TORCH_CHECK(q.dim() == 3 && out.dim() == 3);
  const int Hq = q.size(1);
  const int D = q.size(2);
  const int B = context_lens.size(0);
  const int block_size = k_cache.size(1);
  const int Hk = k_cache.size(2);
  TORCH_CHECK(k_cache.size(3) == D && Hq % Hk == 0);
  TORCH_CHECK(q.stride(2) == 1 && q.stride(1) == D && out.stride(2) == 1 && out.stride(1) == D);
  TORCH_CHECK(q.stride(0) % 2 == 0 && out.stride(0) % 2 == 0);
  TORCH_CHECK(is_aligned16(k_cache.data_ptr()) && is_aligned16(v_cache.data_ptr()));
  if (B == 0 || max_q_len == 0) return;
  const at::cuda::OptionalCUDAGuard guard(q.device());
  const float scale_log2 = static_cast<float>(scale) * 1.4426950408889634f;
  dim3 grid((max_q_len + kBM - 1) / kBM, Hq, B);
#define LAUNCH(D_)                                                                                               \
  paged_prefill_kernel<D_><<<grid, kThreads, 0, current_stream()>>>(                                             \
      reinterpret_cast<bf16*>(out.data_ptr()), out.stride(0), reinterpret_cast<const bf16*>(q.data_ptr()),       \
      q.stride(0), reinterpret_cast<const bf16*>(k_cache.data_ptr()),                                            \
      reinterpret_cast<const bf16*>(v_cache.data_ptr()), block_tables.data_ptr<int>(), block_tables.size(1),     \
      cu_seqlens_q.data_ptr<int>(), context_lens.data_ptr<int>(), block_size, Hk, Hq / Hk, scale_log2)
  switch (D) {
    case 64: LAUNCH(64); break;
    case 128: LAUNCH(128); break;
    default: TORCH_CHECK(false, "unsupported head_dim ", D);
  }
#undef LAUNCH
}
