"""Complete TP4 scheme benchmark WITH request_indices (scattered, zero gather).

BS in {64,128,256,512}, HV=16/H=4 (TP4), 33% flush, subset launches run on
SCATTERED rows of the full interleaved ring via request_indices. CUPTI cold-L2.

  out16_all : wy_output_only, all B (verify baseline / (c) critical path)
  a_mixed   : ONE kernel-B launch, all B, 33% flush_steps>0 (option a)
  fl_ind    : kernel B fused on the 33% — SCATTERED via request_indices
  ln_kB_ind : kernel B (flush_steps=0) on the 67% — SCATTERED (lean stand-in;
              wy_output_only has no indirection yet, so this is the zero-
              packing lean option available today)
  b_ind     : graphed pair fl_ind + ln_kB_ind  (split, zero packing)
  so_ind    : state-only fold on the 33% — SCATTERED ((c) draft-window fold,
              zero gather)
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
torch.set_grad_enabled(False)


def t_us(fn, graph=False):
    return np.median(bench_gpu_time(fn, dry_run_iters=10, repeat_iters=300,
                                    enable_cupti=True, use_cuda_graph=graph,
                                    cold_l2_cache=True)) * 1000


def main():
    print(f"GPU: {torch.cuda.get_device_name()}  HV={HV} H={H} (TP4), 33% "
          f"flush, SCATTERED subsets via request_indices, CUPTI cold-L2\n")
    hdr = (f"{'BS':>4} | {'out16_all':>9} {'a_mixed':>8} | {'fl_ind':>7} "
           f"{'ln_kB_ind':>9} {'b_ind':>7} | {'so_ind':>7}")
    print(hdr)
    print("-" * len(hdr))
    for B in (64, 128, 256, 512):
        n_f = (B * 33 + 99) // 100
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
            idx_all = torch.arange(B, dtype=torch.int32)
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(B, generator=g)
        fl = perm[:n_f].sort().values.to(torch.int32).to("cuda")
        ln = perm[n_f:].sort().values.to(torch.int32).to("cuda")
        p_mix = torch.zeros(B, dtype=torch.int32, device="cuda")
        p_mix[fl.long()] = 12
        p_fl = torch.full((n_f,), 12, dtype=torch.int32, device="cuda")
        p_ln0 = torch.zeros(B - n_f, dtype=torch.int32, device="cuda")
        common = dict(A_log=A_log, dt_bias=dt_bias,
                      use_qk_l2norm_in_kernel=True, scale=SCALE)

        out16 = t_us(lambda: wy_out(**tok, **common,
                                    initial_state_source=state,
                                    initial_state_indices=idx_all,
                                    disable_state_update=True))
        a_mixed = t_us(lambda: wy_flush(**tok, **common,
                                        initial_state_source=state,
                                        flush_steps=p_mix,
                                        disable_state_update=False))
        fl_ind = t_us(lambda: wy_flush(**tok, **common,
                                       initial_state_source=state,
                                       request_indices=fl, flush_steps=p_fl,
                                       disable_state_update=False))
        ln_ind = t_us(lambda: wy_flush(**tok, **common,
                                       initial_state_source=state,
                                       request_indices=ln, flush_steps=p_ln0,
                                       disable_state_update=False))

        def b_pair():
            wy_flush(**tok, **common, initial_state_source=state,
                     request_indices=fl, flush_steps=p_fl,
                     disable_state_update=False)
            wy_flush(**tok, **common, initial_state_source=state,
                     request_indices=ln, flush_steps=p_ln0,
                     disable_state_update=False)

        b_ind = t_us(b_pair, graph=True)
        so_ind = t_us(lambda: wy_flush(**tok, **common,
                                       initial_state_source=state,
                                       request_indices=fl, flush_steps=p_fl,
                                       disable_state_update=False,
                                       disable_output=True))
        print(f"{B:>4} | {out16:9.2f} {a_mixed:8.2f} | {fl_ind:7.2f} "
              f"{ln_ind:9.2f} {b_ind:7.2f} | {so_ind:7.2f}", flush=True)


if __name__ == "__main__":
    main()
