"""Probe flush-kernel time vs GDN_WY_MBP (launch bounds) at f=0 and f=1."""

import math

import numpy as np
import torch

from flashinfer.gdn_kernels.gdn_decode_bf16_wy_state_and_output import (
    gated_delta_rule_mtp as wy_flush,
)
from flashinfer.testing import bench_gpu_time

B, H, HV, K, V = 256, 16, 64, 128, 128
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
    idx = torch.arange(B, dtype=torch.int32)
    p0 = torch.zeros(B, dtype=torch.int32)
    p1 = torch.full((B,), 12, dtype=torch.int32)

for tag, p in (("f=0", p0), ("f=1", p1)):
    us = (
        np.median(
            bench_gpu_time(
                lambda: wy_flush(
                    **tok, A_log=A_log, dt_bias=dt_bias,
                    initial_state_source=state, initial_state_indices=idx,
                    flush_steps=p, disable_state_update=False,
                    use_qk_l2norm_in_kernel=True, scale=1 / math.sqrt(128),
                ),
                dry_run_iters=10, repeat_iters=300,
                enable_cupti=True, cold_l2_cache=True,
            )
        )
        * 1000
    )
    import os
    print(f"mbp={os.environ.get('GDN_WY_MBP', 'def')} {tag}: {us:.2f} us", flush=True)
