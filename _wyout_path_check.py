"""wy_output_only alternate production paths vs references (report-only).
Env decides the path (flags are read at import):
  default            -> native T in {4,8}, staged otherwise
  SGLANG_GDN_WY_NATIVE_T=0 -> forced staging for all T<16
  SGLANG_GDN_WY_STRIDED_QKV=1 -> strided conv-slice reads (native T only)
Checks T in {2,3,4,5,8,15,16} x HV in {32,64} x 2 seeds against the recurrent
kernel (<=2e-3) and, for the strided path, BIT-equality vs contiguous.
"""
import math
import os
import sys

import torch

from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
    gated_delta_rule_mtp as recurrent,
)
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_output_only import (
    gated_delta_rule_mtp as wy_out,
)

K = V = 128
SCALE = 1.0 / math.sqrt(K)
DEV = "cuda"
STRIDED = os.environ.get("SGLANG_GDN_WY_STRIDED_QKV", "0") != "0"
fails = 0


def run(HV, H, B, Tt, seed):
    global fails
    g = torch.Generator(device=DEV).manual_seed(seed)

    def rn(*s, sc=1.0):
        return torch.randn(*s, generator=g, dtype=torch.bfloat16, device=DEV) * sc

    if STRIDED:
        if Tt not in (4, 8):
            return
        conv_dim = H * K + H * K + HV * V
        mixed = rn(B, Tt, conv_dim)
        q = mixed[:, :, : H * K].view(B, Tt, H, K)
        k = mixed[:, :, H * K : 2 * H * K].view(B, Tt, H, K)
        v = mixed[:, :, 2 * H * K :].view(B, Tt, HV, V)
    else:
        q = rn(B, Tt, H, K)
        k = rn(B, Tt, H, K)
        v = rn(B, Tt, HV, V)
    a = rn(B, Tt, HV, sc=0.1)
    bb = rn(B, Tt, HV)
    A_log = torch.randn(HV, generator=g, dtype=torch.float32, device=DEV) * 0.1
    dtb = torch.randn(HV, generator=g, dtype=torch.float32, device=DEV) * 0.1
    st = torch.randn(B, HV, V, K, generator=g, dtype=torch.bfloat16, device=DEV)
    idx = torch.arange(B, dtype=torch.int32, device=DEV)
    kw = dict(A_log=A_log, a=a, dt_bias=dtb, b=bb,
              initial_state_indices=idx, disable_state_update=True,
              use_qk_l2norm_in_kernel=True, scale=SCALE)
    o = wy_out(q=q, k=k, v=v, initial_state_source=st.clone(), **kw)
    o_rec = recurrent(q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),
                      initial_state_source=st.clone(), **kw)
    torch.cuda.synchronize()
    d = (o.float() - o_rec.float()).abs().max().item()
    tag = f"T={Tt} HV={HV} B={B} s={seed}"
    if d > 2e-3:
        print(f"FAIL {tag}: vs recurrent {d:.2e}", flush=True)
        fails += 1
    if STRIDED:
        oc = wy_out(q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),
                    initial_state_source=st.clone(), **kw)
        torch.cuda.synchronize()
        if not torch.equal(o, oc):
            print(f"FAIL {tag}: strided != contiguous (bit)", flush=True)
            fails += 1


def main():
    torch.set_grad_enabled(False)
    mode = ("STRIDED" if STRIDED else
            ("FORCED-STAGING" if os.environ.get("SGLANG_GDN_WY_NATIVE_T") == "0"
             else "DEFAULT"))
    print(f"wy_output_only path check [{mode}]", flush=True)
    for HV, H in ((32, 16), (64, 16)):
        for Tt in (2, 3, 4, 5, 8, 15, 16):
            for B in (1, 33):
                for seed in (0, 1):
                    run(HV, H, B, Tt, seed)
        print(f"  HV={HV} done", flush=True)
    print(f"{'PASS' if fails == 0 else str(fails) + ' FAIL(S)'} [{mode}]")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
