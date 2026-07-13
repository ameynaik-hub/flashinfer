"""State-only fold (disable_output=True) vs fused flush vs branch recovery.

The state-only number is what an overlapped-with-draft-phase flush would ask
the draft window to absorb. B=256; HV=64 (single GPU) and HV=16/H=4 (TP4).

Run: source env.sh && python _bench_state_only.py
"""

import math

import numpy as np
import torch

from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
    gated_delta_rule_mtp as branch_mtp,
)
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_state_and_output import (
    gated_delta_rule_mtp as wy_flush,
)
from flashinfer.testing import bench_gpu_time

B, K_DIM, V_DIM = 256, 128, 128
SCALE = 1.0 / math.sqrt(K_DIM)


def run(HV, H):
    with torch.device("cuda"):
        tok = dict(
            q=torch.randn(B, 16, H, K_DIM, dtype=torch.bfloat16),
            k=torch.randn(B, 16, H, K_DIM, dtype=torch.bfloat16),
            v=torch.randn(B, 16, HV, V_DIM, dtype=torch.bfloat16),
            a=torch.randn(B, 16, HV, dtype=torch.bfloat16) * 0.1,
            b=torch.randn(B, 16, HV, dtype=torch.bfloat16),
        )
        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
        state = torch.randn(B, HV, V_DIM, K_DIM, dtype=torch.bfloat16)
        idx = torch.arange(B, dtype=torch.int32)
        p12 = torch.full((B,), 12, dtype=torch.int32)
        acc13 = torch.full((B,), 12, dtype=torch.int32)
    common = dict(A_log=A_log, dt_bias=dt_bias, initial_state_indices=idx,
                  use_qk_l2norm_in_kernel=True, scale=SCALE)

    def t_us(fn):
        return (
            np.median(
                bench_gpu_time(fn, dry_run_iters=10, repeat_iters=300,
                               enable_cupti=True, cold_l2_cache=True)
            )
            * 1000
        )

    so = t_us(lambda: wy_flush(**tok, **common, initial_state_source=state,
                               flush_steps=p12, disable_state_update=False,
                               disable_output=True))
    fused = t_us(lambda: wy_flush(**tok, **common, initial_state_source=state,
                                  flush_steps=p12, disable_state_update=False))
    br = t_us(lambda: branch_mtp(**tok, **common, initial_state_source=state,
                                 accepted_steps=acc13,
                                 disable_state_update=False,
                                 disable_output=True))
    print(f"HV={HV:>2} H={H:>2} B={B}: state-only fold {so:7.2f} us | "
          f"fused flush {fused:7.2f} | branch recovery {br:7.2f}", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    print(f"GPU: {torch.cuda.get_device_name()}")
    run(64, 16)
    run(16, 4)
