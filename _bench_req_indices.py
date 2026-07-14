"""request_indices payoff at TP4 (HV=16, H=4): subset launches on SCATTERED
rows of a B=256 interleaved ring vs packed sub-batch. n_sub=85 (33%), P=12.
CUPTI cold-L2, median of 300."""
import math
import numpy as np
import torch
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_state_and_output import (
    gated_delta_rule_mtp as wy_flush,
)
from flashinfer.testing import bench_gpu_time

H, HV, K, V = 4, 16, 128, 128
B, N = 256, 85
SCALE = 1.0 / math.sqrt(K)
torch.set_grad_enabled(False)
with torch.device("cuda"):
    tok = dict(
        q=torch.randn(B, 16, H, K, dtype=torch.bfloat16),
        k=torch.randn(B, 16, H, K, dtype=torch.bfloat16),
        v=torch.randn(B, 16, HV, V, dtype=torch.bfloat16),
        a=torch.randn(B, 16, HV, dtype=torch.bfloat16) * 0.1,
        b=torch.randn(B, 16, HV, dtype=torch.bfloat16),
    )
    A_log = torch.randn(HV, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
    state = torch.randn(B, HV, V, K, dtype=torch.bfloat16)
g = torch.Generator().manual_seed(0)
sub = torch.randperm(B, generator=g)[:N].sort().values.to(torch.int32).to("cuda")
sl = sub.long()
p_f = torch.full((N,), 12, dtype=torch.int32, device="cuda")
tok_p = {k2: v2[sl].contiguous() for k2, v2 in tok.items()}
common = dict(A_log=A_log, dt_bias=dt_bias, use_qk_l2norm_in_kernel=True,
              scale=SCALE)

def t_us(fn):
    return np.median(bench_gpu_time(fn, dry_run_iters=10, repeat_iters=300,
                                    enable_cupti=True, cold_l2_cache=True)) * 1000

fused_ind = t_us(lambda: wy_flush(**tok, **common, initial_state_source=state,
                                  request_indices=sub, flush_steps=p_f,
                                  disable_state_update=False))
so_ind = t_us(lambda: wy_flush(**tok, **common, initial_state_source=state,
                               request_indices=sub, flush_steps=p_f,
                               disable_state_update=False, disable_output=True))
fused_pk = t_us(lambda: wy_flush(**tok_p, **common, initial_state_source=state,
                                 initial_state_indices=sub, flush_steps=p_f,
                                 disable_state_update=False))
so_pk = t_us(lambda: wy_flush(**tok_p, **common, initial_state_source=state,
                              initial_state_indices=sub, flush_steps=p_f,
                              disable_state_update=False, disable_output=True))
print(f"GPU: {torch.cuda.get_device_name()}  B={B} sub={N} HV={HV} (TP4)")
print(f"fused  kernel-B on 85: scattered-indirect {fused_ind:6.2f} us | "
      f"packed {fused_pk:6.2f} us  (packed needed gather ~11 us on top)")
print(f"state-only fold on 85: scattered-indirect {so_ind:6.2f} us | "
      f"packed {so_pk:6.2f} us  (packed needed gather ~11 us on top)")
