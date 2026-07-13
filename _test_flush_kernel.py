"""Correctness test for gdn_decode_bf16_wy_state_and_output (Phase-5 flush).

Per regime x batch:
 1. OUTPUT bit-equality vs gdn_decode_bf16_wy_output_only (Phases 1-4 untouched).
 2. STATE: folded pool rows vs (a) branch-kernel accepted_steps fold (production
    oracle) and (b) torch fp32/bf16 sequential references (ground truth).
 3. flush_steps == 0 -> that request's state stays bit-unchanged.

Run (B200, from repo root): source env.sh && python _test_flush_kernel.py
"""

import math
import os
import sys

import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "tests", "gdn"))
from reference_delta_rule import decode_delta_rule  # noqa: E402

from flashinfer.gdn_kernels.gdn_decode_bf16_state import (  # noqa: E402
    gated_delta_rule_mtp as branch_mtp,
)
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_output_only import (  # noqa: E402
    gated_delta_rule_mtp as wy_out,
)
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_state_and_output import (  # noqa: E402
    gated_delta_rule_mtp as wy_flush,
)

H, HV, K_DIM, V_DIM, T = 16, 64, 128, 128, 16
SCALE = 1.0 / math.sqrt(K_DIM)
REGIMES = {
    "mild": (0.1, 0.1, 0.1, 1.0),
    "strong": (1.0, 1.0, 1.0, 1.0),
    "big": (0.1, 0.1, 0.1, 30.0),
}


def run(B, regime, seed=0):
    sA, sdt, sa, sS = REGIMES[regime]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(B, T, H, K_DIM, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, K_DIM, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, HV, V_DIM, dtype=torch.bfloat16, device=dev)
    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device=dev) * sa
    bb = torch.randn(B, T, HV, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(HV, dtype=torch.float32, device=dev) * sA
    dt_bias = torch.randn(HV, dtype=torch.float32, device=dev) * sdt
    S0 = torch.randn(B, HV, V_DIM, K_DIM, dtype=torch.bfloat16, device=dev) * sS
    idx = torch.arange(B, dtype=torch.int32, device=dev)
    P = torch.randint(1, 13, (B,), dtype=torch.int32, device=dev)
    P[0] = 0  # per-request predication check
    if B > 1:
        P[1] = 12
    if B > 2:
        P[2] = 1

    common = dict(
        A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=bb,
        initial_state_indices=idx, use_qk_l2norm_in_kernel=True, scale=SCALE,
    )

    o_ref = wy_out(
        initial_state_source=S0.clone(), disable_state_update=True, **common
    )
    S_new = S0.clone()
    o_new = wy_flush(
        initial_state_source=S_new,
        disable_state_update=False,
        flush_steps=P,
        **common,
    )
    torch.cuda.synchronize()

    bit_out = torch.equal(o_new, o_ref)
    p0_unchanged = torch.equal(S_new[0], S0[0])

    # State-only variant (disable_output=True): the fold must be BIT-identical
    # to the fused kernel's (same math path, output machinery compiled out).
    S_so = S0.clone()
    r_so = wy_flush(
        initial_state_source=S_so,
        disable_state_update=False,
        disable_output=True,
        flush_steps=P,
        **common,
    )
    torch.cuda.synchronize()
    so_bit = torch.equal(S_so, S_new) and r_so is None

    # Production oracle: branch kernel per-request fold of the first P_i rows.
    # accepted_steps needs K_i >= 1; fold 1 row for the P=0 slots and exclude
    # them from the comparison via `mask`.
    S_branch = S0.clone()
    mask = (P > 0).view(B, 1, 1, 1)
    P_b = torch.where(P > 0, P, torch.ones_like(P))
    branch_mtp(
        A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=bb,
        initial_state_source=S_branch, initial_state_indices=idx,
        accepted_steps=(P_b - 1).to(torch.int32),
        disable_state_update=False, disable_output=True,
        use_qk_l2norm_in_kernel=True, scale=SCALE,
    )
    torch.cuda.synchronize()

    # Torch references: fold t rows sequentially, gather per request at P_i.
    def torch_fold(state_dtype):
        state = S0.transpose(-2, -1).contiguous().to(state_dtype)  # [B,HV,K,V]
        states = [state.clone()]
        for t in range(12):
            _, state = decode_delta_rule(
                q[:, t].float(), k[:, t].float(), v[:, t].float(), state,
                A_log=A_log, a=a[:, t], dt_bias=dt_bias, b=bb[:, t],
                scale_factor=SCALE, softplus_beta=1.0, softplus_threshold=20.0,
                use_l2_norm=True, state_dtype=state_dtype,
            )
            states.append(state.clone())
        stk = torch.stack(states, 0)  # [13, B, HV, K, V]
        sel = stk[P.long(), torch.arange(B, device=dev)]  # [B,HV,K,V]
        return sel.transpose(-2, -1).contiguous().float()  # kernel layout

    S32 = torch_fold(torch.float32)
    S16 = torch_fold(torch.bfloat16)

    def md(x, y):
        return (x - y).masked_fill(~mask, 0.0).abs().max().item()

    e_new32 = md(S_new.float(), S32)
    e_br32 = md(S_branch.float(), S32)
    e_1632 = md(S16, S32)
    e_new_br = md(S_new.float(), S_branch.float())
    floor = 1e-3 * max(sS, 1.0)
    ok = (
        bit_out
        and p0_unchanged
        and so_bit
        and e_new32 <= max(2.0 * max(e_br32, e_1632), floor)
    )
    print(
        f"[{regime:>6} B={B:>3}] out_bit={bit_out} P0_unchanged={p0_unchanged} "
        f"stateonly_bit={so_bit} | "
        f"S: new-vs-32 {e_new32:.2e}  br-vs-32 {e_br32:.2e}  "
        f"seq16-vs-32 {e_1632:.2e}  new-vs-br {e_new_br:.2e}"
        f"{'' if ok else '  <-- FAIL'}",
        flush=True,
    )
    return 0 if ok else 1


def main():
    torch.set_grad_enabled(False)
    print(f"GPU: {torch.cuda.get_device_name()}")
    fails = 0
    for regime in REGIMES:
        for B in (1, 8, 64):
            fails += run(B, regime)
    print("ALL PASS" if fails == 0 else f"{fails} FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
