"""Qwen3.5-on-TP4 per-GPU cases (conc=256, HV=64/4=16, H=16/4=4), CUPTI cold-L2.

  case 0a: output-only kernel, ALL 256 requests, T=4 rows   (today's verify)
  case 0b: output-only kernel, ALL 256 requests, 16-row window (replay verify)
  case A : ONE kernel-B launch, ALL 256, 33% flush (85 x flush_steps=12, 171 x 0)
  case B : kernel B on the 85 flushers + output-only(16-row) on the other 171
           (split dispatch; reported eager AND CUDA-graph-captured — the
           graphed number is the honest one at these kernel sizes)

Run: source env.sh && python _bench_tp4_cases.py
"""

import math

import numpy as np
import torch

from flashinfer.gdn_kernels.gdn_decode_bf16_wy_output_only import (
    gated_delta_rule_mtp as wy_out,
)
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_state_and_output import (
    gated_delta_rule_mtp as wy_flush,
)
from flashinfer.testing import bench_gpu_time

B, H, HV, K_DIM, V_DIM = 256, 4, 16, 128, 128
SCALE = 1.0 / math.sqrt(K_DIM)
N_F = 85  # 33% of 256


def tok(n, t):
    with torch.device("cuda"):
        return dict(
            q=torch.randn(n, t, H, K_DIM, dtype=torch.bfloat16),
            k=torch.randn(n, t, H, K_DIM, dtype=torch.bfloat16),
            v=torch.randn(n, t, HV, V_DIM, dtype=torch.bfloat16),
            a=torch.randn(n, t, HV, dtype=torch.bfloat16) * 0.1,
            b=torch.randn(n, t, HV, dtype=torch.bfloat16),
        )


def main():
    torch.set_grad_enabled(False)
    with torch.device("cuda"):
        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
        state = torch.randn(B, HV, V_DIM, K_DIM, dtype=torch.bfloat16)
        idx_all = torch.arange(B, dtype=torch.int32)
    common = dict(A_log=A_log, dt_bias=dt_bias, use_qk_l2norm_in_kernel=True,
                  scale=SCALE)
    tok4 = tok(B, 4)
    tok16 = tok(B, 16)

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(B, generator=g)
    fl_idx = perm[:N_F].sort().values.to("cuda")
    ln_idx = perm[N_F:].sort().values.to("cuda")
    p_mix = torch.zeros(B, dtype=torch.int32, device="cuda")
    p_mix[fl_idx] = 12
    tok_f = {k: v[fl_idx].contiguous() for k, v in tok16.items()}
    tok_l = {k: v[ln_idx].contiguous() for k, v in tok16.items()}
    idx_f = fl_idx.to(torch.int32)
    idx_l = ln_idx.to(torch.int32)
    p_f = torch.full((N_F,), 12, dtype=torch.int32, device="cuda")

    def t_us(fn, graph=False):
        return (
            np.median(
                bench_gpu_time(fn, dry_run_iters=10, repeat_iters=300,
                               enable_cupti=True, use_cuda_graph=graph,
                               cold_l2_cache=True)
            )
            * 1000
        )

    v4 = t_us(lambda: wy_out(**tok4, **common, initial_state_source=state,
                             initial_state_indices=idx_all,
                             disable_state_update=True))
    v16 = t_us(lambda: wy_out(**tok16, **common, initial_state_source=state,
                              initial_state_indices=idx_all,
                              disable_state_update=True))
    single = t_us(lambda: wy_flush(**tok16, **common,
                                   initial_state_source=state,
                                   initial_state_indices=idx_all,
                                   flush_steps=p_mix,
                                   disable_state_update=False))
    fl = t_us(lambda: wy_flush(**tok_f, **common, initial_state_source=state,
                               initial_state_indices=idx_f, flush_steps=p_f,
                               disable_state_update=False))
    ln = t_us(lambda: wy_out(**tok_l, **common, initial_state_source=state,
                             initial_state_indices=idx_l,
                             disable_state_update=True))

    def split_fn():
        wy_flush(**tok_f, **common, initial_state_source=state,
                 initial_state_indices=idx_f, flush_steps=p_f,
                 disable_state_update=False)
        wy_out(**tok_l, **common, initial_state_source=state,
               initial_state_indices=idx_l, disable_state_update=True)

    split_eager = t_us(split_fn)
    split_graph = t_us(split_fn, graph=True)

    print(f"GPU: {torch.cuda.get_device_name()}  conc={B} HV={HV} H={H} (TP4)")
    print(f"case 0a  output-only, all 256, T=4 rows        : {v4:7.2f} us")
    print(f"case 0b  output-only, all 256, 16-row window   : {v16:7.2f} us")
    print(f"case A   single kernel-B, all 256, 33% flush   : {single:7.2f} us")
    print(f"case B   split: kernel-B(85) + output-only(171): "
          f"{split_graph:7.2f} us graphed  ({fl:.2f} + {ln:.2f}; "
          f"eager {split_eager:.2f} incl. host gap)")


if __name__ == "__main__":
    main()
