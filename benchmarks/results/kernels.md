GPU: NVIDIA GeForce RTX 5070 Ti, torch 2.14.1+cu130, CUDA 13.0

### Paged decode attention (Hq=16, Hk=8, D=128, block=16, bf16)

| batch | ctx | ours (us) | ours GB/s | gather+SDPA (us) | contiguous SDPA (us) | vs gather | vs contiguous |
|---|---|---|---|---|---|---|---|
| 1 | 1024 | 20.8 | 202 | 31.1 | 17.2 | 1.5x | 0.83x |
| 1 | 4096 | 31.1 | 539 | 82.8 | 34.8 | 2.7x | 1.12x |
| 1 | 16384 | 97.1 | 691 | 276.8 | 98.6 | 2.9x | 1.02x |
| 8 | 2048 | 95.1 | 706 | 270.1 | 95.4 | 2.8x | 1.00x |
| 32 | 1024 | 183.1 | 733 | 593.3 | 218.2 | 3.2x | 1.19x |
| 32 | 4096 | 680.8 | 789 | 2284.8 | 827.8 | 3.4x | 1.22x |
| 64 | 2048 | 680.4 | 789 | 2288.8 | 838.4 | 3.4x | 1.23x |
| 128 | 1024 | 682.3 | 787 | 2246.1 | 803.1 | 3.3x | 1.18x |

### Causal prefill attention (paged K/V vs PyTorch SDPA on contiguous K/V)

| batch | seqlen | ours (us) | ours TFLOPS | SDPA (us) | SDPA TFLOPS | speedup |
|---|---|---|---|---|---|---|
| 8 | 512 | 181 | 47.6 | 160 | 53.7 | 0.89x |
| 4 | 1024 | 311 | 55.2 | 269 | 64.0 | 0.86x |
| 2 | 2048 | 573 | 59.9 | 490 | 70.2 | 0.85x |
| 1 | 4096 | 1094 | 62.8 | 929 | 74.0 | 0.85x |
| 1 | 8192 | 3999 | 68.7 | 3343 | 82.2 | 0.84x |

### Fused memory-bound kernels vs eager PyTorch

| kernel | ours (us) | ours GB/s | eager (us) | speedup |
|---|---|---|---|---|
| rms_norm [4096x2048] | 45.3 | 740 | 264.5 | 5.8x |
| fused_add_rms_norm [4096x2048] | 88.2 | 761 | 306.0 | 3.5x |
| silu_and_mul [4096x2x6144] | 195.1 | 774 | 355.1 | 1.8x |
| qk_norm_rope [4096 tok, 16q+8k heads] | 69.9 | 720 | 1026.9 | 14.7x |
