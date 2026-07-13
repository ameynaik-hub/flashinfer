"""Cost of the flush-time ring gather for the sub-batch state-only fold (TP4).

Gathers k/v/a/b rings ([B,16,...] -> [n_f,16,...], n_f = 33% of B, random rows)
into preallocated scratch. Eager and CUDA-graphed. q is not gathered (unread
by the state-only kernel).
"""
import numpy as np
import torch
from flashinfer.testing import bench_gpu_time

H, HV, K, V = 4, 16, 128, 128
torch.set_grad_enabled(False)
print(f"GPU: {torch.cuda.get_device_name()}  HV={HV} (TP4) ring gather, 33%")
for B in (64, 128, 256, 512):
    n_f = (B * 33 + 99) // 100
    with torch.device("cuda"):
        rk = torch.randn(B, 16, H, K, dtype=torch.bfloat16)
        rv = torch.randn(B, 16, HV, V, dtype=torch.bfloat16)
        ra = torch.randn(B, 16, HV, dtype=torch.bfloat16)
        rb = torch.randn(B, 16, HV, dtype=torch.bfloat16)
        ok = torch.empty(n_f, 16, H, K, dtype=torch.bfloat16)
        ov = torch.empty(n_f, 16, HV, V, dtype=torch.bfloat16)
        oa = torch.empty(n_f, 16, HV, dtype=torch.bfloat16)
        ob = torch.empty(n_f, 16, HV, dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(B, generator=g)[:n_f].sort().values.to("cuda")

    def gather():
        torch.index_select(rk, 0, idx, out=ok)
        torch.index_select(rv, 0, idx, out=ov)
        torch.index_select(ra, 0, idx, out=oa)
        torch.index_select(rb, 0, idx, out=ob)

    def t(fn, graph):
        return np.median(bench_gpu_time(
            fn, dry_run_iters=10, repeat_iters=300, enable_cupti=True,
            use_cuda_graph=graph, cold_l2_cache=True)) * 1000

    mb = (rk[0].numel() + rv[0].numel() + ra[0].numel() + rb[0].numel()) * 2 * n_f / 1e6
    print(f"B={B:>4} n_f={n_f:>3} ({mb:5.1f} MB moved): "
          f"eager {t(gather, False):6.2f} us | graphed {t(gather, True):6.2f} us",
          flush=True)
