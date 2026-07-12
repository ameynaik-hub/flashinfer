"""Dispatch-policy bench at 50% flush (B=256): one mixed kernel-B launch vs
splitting the batch (kernel B for the flushing half, output-only for the rest).

Modes (all CUPTI cold-L2, per-iteration total kernel time):
  single   : ONE wy_state_and_output launch, B=256, flush_steps = 12 for the
             flush half, 0 for the lean half (v1.5 predication).
  split    : wy_state_and_output on a PRE-PACKED B=128 flush sub-batch (all
             P=12) + wy_output_only on a pre-packed B=128 lean sub-batch.
             State pool is shared via initial_state_indices (no state gather).
             Assumes serving writes rings pre-grouped (group membership is
             known before draft rows are written).
  split+gth: same two launches, but the 5 input tensors for both halves are
             index_select-gathered from the interleaved B=256 rings every
             iteration (the naive retrofit cost).

Also sweeps the flush fraction f to find the policy crossover.

Run: source env.sh && python _bench_split_dispatch.py
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

H, HV, K_DIM, V_DIM = 16, 64, 128, 128  # H/HV overridable via --H/--HV (TP shards)
SCALE = 1.0 / math.sqrt(K_DIM)


def make_tok(B):
    with torch.device("cuda"):
        return dict(
            q=torch.randn(B, 16, H, K_DIM, dtype=torch.bfloat16),
            k=torch.randn(B, 16, H, K_DIM, dtype=torch.bfloat16),
            v=torch.randn(B, 16, HV, V_DIM, dtype=torch.bfloat16),
            a=torch.randn(B, 16, HV, dtype=torch.bfloat16) * 0.1,
            b=torch.randn(B, 16, HV, dtype=torch.bfloat16),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--HV", type=int, default=64, help="v-heads per GPU (TP-sharded)")
    ap.add_argument("--H", type=int, default=16, help="q/k heads per GPU (TP-sharded)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    B = args.B
    global H, HV
    H, HV = args.H, args.HV

    with torch.device("cuda"):
        A_log = torch.randn(HV, dtype=torch.float32) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
        state = torch.randn(B, HV, V_DIM, K_DIM, dtype=torch.bfloat16)
    tok = make_tok(B)
    common = dict(A_log=A_log, dt_bias=dt_bias, use_qk_l2norm_in_kernel=True,
                  scale=SCALE)

    def t_us(fn):
        return (
            np.median(
                bench_gpu_time(fn, dry_run_iters=args.warmup,
                               repeat_iters=args.iters, enable_cupti=True,
                               cold_l2_cache=True)
            )
            * 1000
        )

    # A1 reference: today's per-iteration cost = WY verify(T=4) + branch fold(4).
    tok4 = {k: (v[:, :4].contiguous() if v.dim() > 2 else v)
            for k, v in make_tok(B).items()}
    idx_all0 = torch.arange(B, dtype=torch.int32, device="cuda")
    acc4 = torch.full((B,), 3, dtype=torch.int32, device="cuda")
    a1 = t_us(lambda: wy_out(
        **tok4, **common, initial_state_source=state,
        initial_state_indices=idx_all0, disable_state_update=True,
    )) + t_us(lambda: branch_mtp(
        **tok4, **common, initial_state_source=state,
        initial_state_indices=idx_all0, accepted_steps=acc4,
        disable_state_update=False, disable_output=True,
    ))

    print(
        f"GPU: {torch.cuda.get_device_name()}  B={B} HV={HV} H={H} bf16, "
        f"CUPTI cold-L2 median of {args.iters} | A1 (verify4+fold4) = {a1:.2f} us\n"
    )
    hdr = (
        f"{'f%':>4} {'n_fl':>5} | {'single':>8} | {'split':>8} {'(fl+ln)':>15} | "
        f"{'split+gth':>9} | best"
    )
    print(hdr)
    print("-" * len(hdr))
    g = torch.Generator().manual_seed(0)
    for f_pct in (10, 20, 30, 33, 40, 50, 60, 80):
        n_f = (f_pct * B + 99) // 100
        perm = torch.randperm(B, generator=g)
        fl_idx = perm[:n_f].sort().values.to("cuda")
        ln_idx = perm[n_f:].sort().values.to("cuda")

        # --- single mixed launch
        p_mix = torch.zeros(B, dtype=torch.int32, device="cuda")
        p_mix[fl_idx] = 12
        idx_all = torch.arange(B, dtype=torch.int32, device="cuda")
        single = t_us(lambda: wy_flush(
            **tok, **common, initial_state_source=state,
            initial_state_indices=idx_all, flush_steps=p_mix,
            disable_state_update=False))

        # --- split, pre-packed sub-batches (state shared via indices)
        tok_f = {k: v[fl_idx].contiguous() for k, v in tok.items()}
        tok_l = {k: v[ln_idx].contiguous() for k, v in tok.items()}
        idx_f = fl_idx.to(torch.int32)
        idx_l = ln_idx.to(torch.int32)
        p_f = torch.full((n_f,), 12, dtype=torch.int32, device="cuda")

        def split_fn():
            wy_flush(**tok_f, **common, initial_state_source=state,
                     initial_state_indices=idx_f, flush_steps=p_f,
                     disable_state_update=False)
            wy_out(**tok_l, **common, initial_state_source=state,
                   initial_state_indices=idx_l, disable_state_update=True)

        split = t_us(split_fn)
        fl_only = t_us(lambda: wy_flush(
            **tok_f, **common, initial_state_source=state,
            initial_state_indices=idx_f, flush_steps=p_f,
            disable_state_update=False))
        ln_only = t_us(lambda: wy_out(
            **tok_l, **common, initial_state_source=state,
            initial_state_indices=idx_l, disable_state_update=True))

        # --- split with per-iteration input gather from interleaved rings
        def split_gather_fn():
            tf = {k: v.index_select(0, fl_idx) for k, v in tok.items()}
            tl = {k: v.index_select(0, ln_idx) for k, v in tok.items()}
            wy_flush(**tf, **common, initial_state_source=state,
                     initial_state_indices=idx_f, flush_steps=p_f,
                     disable_state_update=False)
            wy_out(**tl, **common, initial_state_source=state,
                   initial_state_indices=idx_l, disable_state_update=True)

        split_g = t_us(split_gather_fn)

        best = min(("single", single), ("split", split),
                   ("split+gth", split_g), key=lambda x: x[1])[0]
        print(
            f"{f_pct:>4} {n_f:>5} | {single:8.2f} | {split:8.2f} "
            f"({fl_only:6.2f}+{ln_only:6.2f}) | {split_g:9.2f} | {best}",
            flush=True,
        )


if __name__ == "__main__":
    main()
