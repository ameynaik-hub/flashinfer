"""TP4 (HV=16, H=4) scheme comparison at 33% flush rate, BS in {64,128,256,512}.

  out4    : output-only kernel, ALL B, T=4 rows (anchor to serving's
            'output kernel time' column)
  out16   : output-only kernel, ALL B, 16-row replay window (ReplaySSM verify)
  a_fused : ONE kernel-B launch, ALL B, 33% flush (P=12), 67% P=0
  b_split : kernel-B on the 33% sub-batch + output-only(16-row) on the 67%
            (CUDA-graph captured pair = honest number at these sizes)
  c_so_sub: state-only fold, 33% sub-batch      (runs in the DRAFT phase)
  c_so_mix: state-only fold, ALL B mixed 33%    (draft phase, no pre-grouping)
  c crit. path = out16 (100% of requests, no flush in verify)

Run: source env.sh && python _bench_tp4_schemes.py
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

H, HV, K, V = 4, 16, 128, 128
SCALE = 1.0 / math.sqrt(K)


def tok(n, t):
    with torch.device("cuda"):
        return dict(
            q=torch.randn(n, t, H, K, dtype=torch.bfloat16),
            k=torch.randn(n, t, H, K, dtype=torch.bfloat16),
            v=torch.randn(n, t, HV, V, dtype=torch.bfloat16),
            a=torch.randn(n, t, HV, dtype=torch.bfloat16) * 0.1,
            b=torch.randn(n, t, HV, dtype=torch.bfloat16),
        )


def t_us(fn, graph=False):
    return np.median(bench_gpu_time(
        fn, dry_run_iters=10, repeat_iters=300, enable_cupti=True,
        use_cuda_graph=graph, cold_l2_cache=True)) * 1000


def main():
    torch.set_grad_enabled(False)
    print(f"GPU: {torch.cuda.get_device_name()}  HV={HV} H={H} (TP4), "
          f"33% flush, CUPTI cold-L2\n")
    hdr = (f"{'BS':>4} | {'out4':>7} {'out16':>7} | {'a_fused':>8} | "
           f"{'b_split':>8} {'(fl+ln)':>15} | {'c_crit':>7} {'c_so_sub':>8} "
           f"{'c_so_mix':>8}")
    print(hdr)
    print("-" * len(hdr))
    for B in (64, 128, 256, 512):
        n_f = (B * 33 + 99) // 100
        with torch.device("cuda"):
            A_log = torch.randn(HV, dtype=torch.float32) * 0.1
            dt_bias = torch.randn(HV, dtype=torch.float32) * 0.1
            state = torch.randn(B, HV, V, K, dtype=torch.bfloat16)
            idx = torch.arange(B, dtype=torch.int32)
            p_mix = torch.zeros(B, dtype=torch.int32)
        perm = torch.randperm(B, generator=torch.Generator().manual_seed(0))
        p_mix[perm[:n_f].to("cuda")] = 12
        with torch.device("cuda"):
            idx_f = torch.arange(n_f, dtype=torch.int32)
            idx_l = torch.arange(n_f, B, dtype=torch.int32)
            p_f = torch.full((n_f,), 12, dtype=torch.int32)
        common = dict(A_log=A_log, dt_bias=dt_bias,
                      use_qk_l2norm_in_kernel=True, scale=SCALE)
        tok4 = tok(B, 4)
        tok16 = tok(B, 16)
        tok_f = tok(n_f, 16)
        tok_l = tok(B - n_f, 16)

        out4 = t_us(lambda: wy_out(**tok4, **common, initial_state_source=state,
                                   initial_state_indices=idx,
                                   disable_state_update=True))
        out16 = t_us(lambda: wy_out(**tok16, **common,
                                    initial_state_source=state,
                                    initial_state_indices=idx,
                                    disable_state_update=True))
        a_fused = t_us(lambda: wy_flush(**tok16, **common,
                                        initial_state_source=state,
                                        initial_state_indices=idx,
                                        flush_steps=p_mix,
                                        disable_state_update=False))
        fl = t_us(lambda: wy_flush(**tok_f, **common,
                                   initial_state_source=state,
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

        b_split = t_us(split_fn, graph=True)
        c_so_sub = t_us(lambda: wy_flush(**tok_f, **common,
                                         initial_state_source=state,
                                         initial_state_indices=idx_f,
                                         flush_steps=p_f,
                                         disable_state_update=False,
                                         disable_output=True))
        c_so_mix = t_us(lambda: wy_flush(**tok16, **common,
                                         initial_state_source=state,
                                         initial_state_indices=idx,
                                         flush_steps=p_mix,
                                         disable_state_update=False,
                                         disable_output=True))
        print(f"{B:>4} | {out4:7.2f} {out16:7.2f} | {a_fused:8.2f} | "
              f"{b_split:8.2f} ({fl:6.2f}+{ln:6.2f}) | {out16:7.2f} "
              f"{c_so_sub:8.2f} {c_so_mix:8.2f}", flush=True)


if __name__ == "__main__":
    main()
