#include <torch/extension.h>

void rms_norm(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& w, double eps);
void fused_add_rms_norm(torch::Tensor& x, torch::Tensor& residual, const torch::Tensor& w, double eps);
void qk_norm_rope(torch::Tensor& qkv, const torch::Tensor& positions, const torch::Tensor& cos_sin,
                  const std::optional<torch::Tensor>& q_w, const std::optional<torch::Tensor>& k_w,
                  int64_t num_q_heads, int64_t num_k_heads, int64_t head_dim, double eps);
void silu_and_mul(torch::Tensor& out, const torch::Tensor& x);
void reshape_and_cache(const torch::Tensor& k, const torch::Tensor& v, torch::Tensor& k_cache, torch::Tensor& v_cache,
                       const torch::Tensor& slot_mapping);
int64_t decode_partition_size();
void paged_decode_attention(torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& k_cache,
                            const torch::Tensor& v_cache, const torch::Tensor& block_tables,
                            const torch::Tensor& context_lens, double scale, int64_t num_partitions,
                            torch::Tensor& tmp_out, torch::Tensor& tmp_ml);
void paged_prefill_attention(torch::Tensor& out, const torch::Tensor& q, const torch::Tensor& k_cache,
                             const torch::Tensor& v_cache, const torch::Tensor& block_tables,
                             const torch::Tensor& cu_seqlens_q, const torch::Tensor& context_lens, int64_t max_q_len,
                             double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm", &rms_norm);
  m.def("fused_add_rms_norm", &fused_add_rms_norm);
  m.def("qk_norm_rope", &qk_norm_rope);
  m.def("silu_and_mul", &silu_and_mul);
  m.def("reshape_and_cache", &reshape_and_cache);
  m.def("decode_partition_size", &decode_partition_size);
  m.def("paged_decode_attention", &paged_decode_attention);
  m.def("paged_prefill_attention", &paged_prefill_attention);
}
