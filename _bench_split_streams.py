"""(b) split pair: sequential same-stream vs concurrent 2-stream (fork-join),
both CUDA-graph-captured, event-timed (warm-L2; the DELTA is the signal).
TP4 (HV=16,H=4), 33% flush, scattered via request_indices."""
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
print(f"GPU: {torch.cuda.get_device_name()}  (b)-pair seq vs 2-stream, "
      f"graphed event timing\n")
side = torch.cuda.Stream()
ev_fork = torch.cuda.Event()
ev_join = torch.cuda.Event()
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
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(B, generator=g)
    fl = perm[:n_f].sort().values.to(torch.int32).to("cuda")
    ln = perm[n_f:].sort().values.to(torch.int32).to("cuda")
    p_fl = torch.full((n_f,), 12, dtype=torch.int32, device="cuda")
    cm = dict(A_log=A_log, dt_bias=dt_bias, use_qk_l2norm_in_kernel=True,
              scale=SCALE)

    def seq_pair():
        wy_flush(**tok, **cm, initial_state_source=state, request_indices=fl,
                 flush_steps=p_fl, disable_state_update=False)
        wy_out(**tok, **cm, initial_state_source=state, request_indices=ln,
               disable_state_update=True)

    def con_pair():
        cur = torch.cuda.current_stream()
        ev_fork.record(cur)
        side.wait_event(ev_fork)
        with torch.cuda.stream(side):
            wy_flush(**tok, **cm, initial_state_source=state,
                     request_indices=fl, flush_steps=p_fl,
                     disable_state_update=False)
            ev_join.record(side)
        wy_out(**tok, **cm, initial_state_source=state, request_indices=ln,
               disable_state_update=True)
        cur.wait_event(ev_join)

    def t(fn):
        return np.median(bench_gpu_time(fn, dry_run_iters=10, repeat_iters=300,
                                        enable_cupti=False, use_cuda_graph=True,
                                        cold_l2_cache=True)) * 1000

    seq = t(seq_pair)
    con = t(con_pair)
    print(f"BS={B:>4}: sequential {seq:7.2f} us | 2-stream {con:7.2f} us | "
          f"speedup {seq / con:5.2f}x", flush=True)
