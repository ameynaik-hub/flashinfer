"""ReplaySSM step 3: kernel-level A/B of the full decode iteration (CUPTI, cold L2).

Measures the five state-kernel primitives a GDN spec-decode iteration is built
from, then composes the per-iteration cost of each scheme:

  primitives (all from a bf16 [B,HV,V,K] state pool, HV=64, H=HK=16, K=V=128):
    verify4  : WY output-only, T=4 draft rows          (1 state read)
    fold4    : branch recovery, fold 4 rows            (1 state read + 1 write)
    fused8   : branch fused recovery_steps=4 + 4 draft (1 state read + 1 write)
    replay16 : WY output-only, full 16-row ring        (1 state read)
    fold13   : branch recovery, fold 13 ring rows      (1 state read + 1 write)

  schemes (per decode iteration, T=4 drafts):
    A1 (verify + separate fold, every iter) : verify4 + fold4
    A2 (fused single call, every iter)      : fused8
    RS (ReplaySSM: replay + amortized flush): replay16 + fold13 / cadence

  cadence = iterations between flushes = (WINDOW - T) / E[accepted] = 12 / 2.5
  = 4.8 by formula; step 2 measured 4 flushes / 18 iters = 4.5. Default 4.5
  (pessimistic), settable via --cadence.

Timing methodology = the #3908 benchmark exactly: CUPTI per-kernel GPU time,
L2 flushed before every iteration, median. verify4/replay16 should reproduce
the PR table's T=4/T=16 columns (cross-check).

Run (B200, from repo root):
    source env.sh && python _replay_step3_bench.py [--iters 500]
"""

import argparse
import math

import numpy as np
import torch

from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
    gated_delta_rule_mtp as branch_mtp,
)
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_output_only import (
    gated_delta_rule_mtp as wy_mtp,
)
from flashinfer.testing import bench_gpu_time

H, HV, K_DIM, V_DIM = 16, 64, 128, 128
WINDOW = 16
T_DRAFT = 4
SCALE = 1.0 / math.sqrt(K_DIM)


def make_tokens(B, T):
    # gating scaled by 0.1 like the #3908 bench: magnitude does not affect
    # timing, but std-1 gating drives repeated folds unstable (inf/nan spam).
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
    ap.add_argument(
        "--batch-size", type=int, nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--cadence", type=float, default=4.5,
                    help="iterations between ReplaySSM flushes (step 2: 4.5)")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    print(
        f"GPU: {torch.cuda.get_device_name()}  HV={HV} H=HK={H} K=V={K_DIM} bf16\n"
        f"CUPTI per-kernel GPU time, cold L2, median of {args.iters} iters, "
        f"warmup {args.warmup}, flush cadence {args.cadence}\n"
    )
    hdr = (
        f"{'B':>4} | {'verify4':>9} {'fold4':>9} {'fused8':>9} {'replay16':>9} "
        f"{'fold13':>9} | {'A1/iter':>9} {'A2/iter':>9} {'RS/iter':>9} | "
        f"{'RSvsA1':>7} {'RSvsA2':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    for B in args.batch_size:
        with torch.device("cuda"):
            A_log = torch.randn(HV, dtype=torch.float32) * 0.1
            dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
            state = torch.randn(B, HV, V_DIM, K_DIM, dtype=torch.bfloat16)
            idx = torch.arange(B, dtype=torch.int32)
            acc4 = torch.full((B,), T_DRAFT - 1, dtype=torch.int32)  # fold 4 rows
            acc13 = torch.full((B,), 12, dtype=torch.int32)  # fold 13 ring rows
        tok4 = make_tokens(B, T_DRAFT)
        tok8 = make_tokens(B, 2 * T_DRAFT)
        tok16 = make_tokens(B, WINDOW)
        common = dict(
            A_log=A_log, dt_bias=dt_bias, initial_state_indices=idx,
            use_qk_l2norm_in_kernel=True, scale=SCALE,
        )

        def t_us(fn):
            return (
                np.median(
                    bench_gpu_time(
                        fn,
                        dry_run_iters=args.warmup,
                        repeat_iters=args.iters,
                        enable_cupti=True,
                        cold_l2_cache=True,
                    )
                )
                * 1000
            )

        verify4 = t_us(lambda: wy_mtp(
            **tok4, **common, initial_state_source=state,
            disable_state_update=True,
        ))
        replay16 = t_us(lambda: wy_mtp(
            **tok16, **common, initial_state_source=state,
            disable_state_update=True,
        ))
        fold4 = t_us(lambda: branch_mtp(
            **tok4, **common, initial_state_source=state,
            accepted_steps=acc4,
            disable_state_update=False, disable_output=True,
        ))
        fold13 = t_us(lambda: branch_mtp(
            **tok16, **common, initial_state_source=state,
            accepted_steps=acc13,
            disable_state_update=False, disable_output=True,
        ))
        try:
            fused8 = t_us(lambda: branch_mtp(
                **tok8, **common, initial_state_source=state,
                recovery_steps=T_DRAFT,
                disable_state_update=False, disable_output=False,
            ))
        except AssertionError:
            fused8 = float("nan")  # fused path needs B*HV >= 128 (wide_vec)

        a1 = verify4 + fold4
        a2 = fused8
        rs = replay16 + fold13 / args.cadence
        print(
            f"{B:>4} | {verify4:9.2f} {fold4:9.2f} {fused8:9.2f} {replay16:9.2f} "
            f"{fold13:9.2f} | {a1:9.2f} {a2:9.2f} {rs:9.2f} | "
            f"{a1 / rs:6.2f}x {a2 / rs:6.2f}x",
            flush=True,
        )

    print(
        "\nstate sweeps/iter: A1 = 3 (read + read/write), A2 = 2 (read/write), "
        f"RS = 1 + 2/{args.cadence} = {1 + 2 / args.cadence:.2f}"
    )


if __name__ == "__main__":
    main()
