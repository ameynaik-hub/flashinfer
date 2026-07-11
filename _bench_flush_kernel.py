"""CUPTI cold-L2 bench of the Phase-5 flush kernel vs the primitives it replaces.

Primitives (bf16, HV=64, H=HK=16, K=V=128, pool=B):
  verify4  : wy_output_only, T=4                     (hot-path verify)
  verify16 : wy_output_only, 16-row window           (replay verify)
  flush16  : wy_state_and_output, P=12 fold + output (ONE launch)
  fold4    : branch recovery, 4 rows                 (scheme A per-iter fold)
  fold13   : branch recovery, 13 rows                (v1 ReplaySSM flush)

Composed per decode iteration (T=4 drafts, flush cadence 4.5):
  A1     = verify4 + fold4                        (today's best)
  RS_v1  = verify16 + fold13/4.5                  (step-3 zero-kernel scheme)
  RS_kB  = (3.5*verify16 + flush16)/4.5           (fused Phase-5 flush)

Run: source env.sh && python _bench_flush_kernel.py [--iters 500]
"""

import argparse
import math

import numpy as np
import torch

from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
    gated_delta_rule_mtp as branch_mtp,
)
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_output_only import (
    gated_delta_rule_mtp as wy_out,
)
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_state_and_output import (
    gated_delta_rule_mtp as wy_flush,
)
from flashinfer.testing import bench_gpu_time

H, HV, K_DIM, V_DIM = 16, 64, 128, 128
SCALE = 1.0 / math.sqrt(K_DIM)
CADENCE = 4.5


def make(B, T):
    with torch.device("cuda"):
        return dict(
            q=torch.randn(B, T, H, K_DIM, dtype=torch.bfloat16),
            k=torch.randn(B, T, H, K_DIM, dtype=torch.bfloat16),
            v=torch.randn(B, T, HV, V_DIM, dtype=torch.bfloat16),
            a=torch.randn(B, T, HV, dtype=torch.bfloat16) * 0.1,
            b=torch.randn(B, T, HV, dtype=torch.bfloat16),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=500)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    print(
        f"GPU: {torch.cuda.get_device_name()}  HV={HV} K=V={K_DIM} bf16 | "
        f"CUPTI cold-L2 median of {args.iters}, cadence {CADENCE}\n"
    )
    hdr = (
        f"{'B':>4} | {'verify4':>8} {'verify16':>9} {'flush16':>8} {'fold4':>8} "
        f"{'fold13':>8} | {'A1':>8} {'RS_v1':>8} {'RS_kB':>8} | "
        f"{'kBvsA1':>7} {'kBvsV1':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for B in args.batch_size:
        with torch.device("cuda"):
            A_log = torch.randn(HV, dtype=torch.float32) * 0.1
            dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
            state = torch.randn(B, HV, V_DIM, K_DIM, dtype=torch.bfloat16)
            idx = torch.arange(B, dtype=torch.int32)
            p12 = torch.full((B,), 12, dtype=torch.int32)
            acc13 = torch.full((B,), 12, dtype=torch.int32)
            acc4 = torch.full((B,), 3, dtype=torch.int32)
        tok4 = make(B, 4)
        tok16 = make(B, 16)
        common = dict(A_log=A_log, dt_bias=dt_bias, initial_state_indices=idx,
                      use_qk_l2norm_in_kernel=True, scale=SCALE)

        def t_us(fn):
            return (
                np.median(
                    bench_gpu_time(fn, dry_run_iters=args.warmup,
                                   repeat_iters=args.iters, enable_cupti=True,
                                   cold_l2_cache=True)
                )
                * 1000
            )

        verify4 = t_us(lambda: wy_out(
            **tok4, **common, initial_state_source=state,
            disable_state_update=True))
        verify16 = t_us(lambda: wy_out(
            **tok16, **common, initial_state_source=state,
            disable_state_update=True))
        flush16 = t_us(lambda: wy_flush(
            **tok16, **common, initial_state_source=state,
            disable_state_update=False, flush_steps=p12))
        fold4 = t_us(lambda: branch_mtp(
            **tok4, **common, initial_state_source=state,
            accepted_steps=acc4, disable_state_update=False,
            disable_output=True))
        fold13 = t_us(lambda: branch_mtp(
            **tok16, **common, initial_state_source=state,
            accepted_steps=acc13, disable_state_update=False,
            disable_output=True))

        a1 = verify4 + fold4
        rs_v1 = verify16 + fold13 / CADENCE
        rs_kb = (3.5 * verify16 + flush16) / CADENCE
        print(
            f"{B:>4} | {verify4:8.2f} {verify16:9.2f} {flush16:8.2f} "
            f"{fold4:8.2f} {fold13:8.2f} | {a1:8.2f} {rs_v1:8.2f} "
            f"{rs_kb:8.2f} | {a1 / rs_kb:6.2f}x {rs_v1 / rs_kb:6.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
