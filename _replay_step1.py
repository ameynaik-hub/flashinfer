"""ReplaySSM step 1: raw v-cache replay correctness + packing cost (zero kernel changes).

Question: does calling the WY output-only kernel ONCE over [h committed rows | T draft
rows] from a checkpoint state S0 reproduce the answer of the production two-step path
(fold h rows into state sequentially, then verify T draft rows from the updated state)?

Per (regime, B) this script generates ONE 16-token sequence and runs the torch
reference (tests/gdn/reference_delta_rule.decode_delta_rule) twice over it:
  - fp32 state  -> ground truth outputs/states        (states32/outs32)
  - bf16 state  -> production recurrence semantics    (states16/outs16)
caching the state after every token, so every (T draft, h hist) combo is a slice.

Paths compared for draft-row outputs (all start from the same bf16 checkpoint S0):
  o_replay : WY kernel, rows [0:h+T] from S0, hist q rows = k rows (nonzero dummy,
             outputs discarded), slice rows [h:h+T].     <-- the new call pattern
  o_2call  : WY kernel, rows [h:h+T] from S1 = bf16-rounded fold of hist (states16[h]).
  o_branch : branch bf16_state kernel, same inputs as o_2call (production comparator).
  outs16   : torch bf16-state sequential outputs        (production recurrence noise).
  outs32   : torch fp32-state sequential outputs        (ground truth, error basis).

PASS = e_replay is at the same scale as the existing paths' error vs ground truth
(e_2call / e_branch / e_seq16) and no NaN/Inf reaches a draft row. h=0 is a built-in
self-check: o_replay and o_2call are the identical call and must match bit-exactly.

Also measures the per-verify packing cost of the ring-buffer pattern: copying T draft
rows of q/k/v/a/b into rows [h:h+T] of persistent 16-row buffers (hist rows already
resident by construction; q dummy rows written once at accept time).

Run (B200, from repo root):
    source env.sh && python _replay_step1.py [--quick]
"""

import argparse
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
    gated_delta_rule_mtp as wy_mtp,
)

H, HV, K_DIM, V_DIM = 16, 64, 128, 128
N_MAX = 16
SCALE = 1.0 / math.sqrt(K_DIM)

# (A_log_scale, dt_bias_scale, a_scale, state_scale)
REGIMES = {
    "test-mild": (0.1, 0.1, 0.1, 1.0),  # standard test-suite magnitudes
    "strong-decay": (1.0, 1.0, 1.0, 1.0),  # real-decay regime (NaN-fix territory)
    "big-state": (0.1, 0.1, 0.1, 30.0),  # adversarial |H0| (WY noise scales with it)
}
COMBOS = [  # (T_draft, h_hist), h+T <= 16
    (4, 0), (4, 4), (4, 8), (4, 12),
    (8, 0), (8, 4), (8, 8),
    (16, 0),
]


def gen_inputs(B, regime, seed):
    sA, sdt, sa, sS = REGIMES[regime]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dev = torch.device("cuda")
    with dev:
        q = torch.randn(B, N_MAX, H, K_DIM, dtype=torch.bfloat16)
        k = torch.randn(B, N_MAX, H, K_DIM, dtype=torch.bfloat16)
        v = torch.randn(B, N_MAX, HV, V_DIM, dtype=torch.bfloat16)
        a = torch.randn(B, N_MAX, HV, dtype=torch.bfloat16) * sa
        b = torch.randn(B, N_MAX, HV, dtype=torch.bfloat16)
        A_log = torch.randn(HV, dtype=torch.float32) * sA
        dt_bias = torch.randn(HV, dtype=torch.float32) * sdt
        S0 = torch.randn(B, HV, V_DIM, K_DIM, dtype=torch.bfloat16) * sS  # kernel layout
    return dict(q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias, S0=S0)


def torch_sweep(inp, state_dtype):
    """Sequential reference over all N_MAX tokens from S0.

    Returns states[t] = state after t tokens (states[0] = S0, ref layout [B,HV,K,V])
    and outs[t] = output of token t (fp32, [B,HV,V]).
    """
    state = inp["S0"].transpose(-2, -1).contiguous()  # -> ref layout [B,HV,K,V]
    state = state.to(state_dtype)
    states, outs = [state.clone()], []
    for t in range(N_MAX):
        o, state = decode_delta_rule(
            inp["q"][:, t].float(),
            inp["k"][:, t].float(),
            inp["v"][:, t].float(),
            state,
            A_log=inp["A_log"],
            a=inp["a"][:, t],
            dt_bias=inp["dt_bias"],
            b=inp["b"][:, t],
            scale_factor=SCALE,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            use_l2_norm=True,
            state_dtype=state_dtype,
        )
        states.append(state.clone())
        outs.append(o.float())
    return states, outs


def compact(t):
    """Copy to canonical-compact strides. `.contiguous()` is NOT enough here: at
    B=1 a [:, :n] slice still reports is_contiguous()==True (size-1 dims are
    exempt from torch's stride check) while keeping the parent's batch stride,
    which the wrapper's mark_compact_shape_dynamic rejects."""
    return torch.empty(t.shape, dtype=t.dtype, device=t.device).copy_(t)


def kernel_call(fn, inp, rows, S_kernel, q_override=None):
    """Run a kernel over token rows [rows.start:rows.stop] from state S_kernel."""
    B = inp["q"].shape[0]
    idx = torch.arange(B, dtype=torch.int32, device="cuda")
    q = compact((q_override if q_override is not None else inp["q"])[:, rows])
    out = fn(
        A_log=inp["A_log"],
        a=compact(inp["a"][:, rows]),
        dt_bias=inp["dt_bias"],
        q=q,
        k=compact(inp["k"][:, rows]),
        v=compact(inp["v"][:, rows]),
        b=compact(inp["b"][:, rows]),
        initial_state_source=S_kernel.clone(),  # pool [B,HV,V,K]; clone: paranoia only
        initial_state_indices=idx,
        disable_state_update=True,
        use_qk_l2norm_in_kernel=True,
        scale=SCALE,
    )
    torch.cuda.synchronize()
    return out


def maxdiff(x, y):
    return (x.float() - y.float()).abs().max().item()


def run_config(inp, states16, outs32, outs16, T, h):
    n = h + T
    # ground truth + production-recurrence outputs for the draft rows
    o32 = torch.stack(outs32[h : h + T], dim=1)  # [B,T,HV,V] fp32
    o16 = torch.stack(outs16[h : h + T], dim=1)
    # S1 = production bf16 fold of the h hist rows, back to kernel layout [B,HV,V,K]
    S1_kernel = states16[h].transpose(-2, -1).contiguous().to(torch.bfloat16)

    # replay: one WY call over [hist|draft] from S0; hist q rows <- k rows (dummy)
    q_replay = inp["q"].clone()
    if h > 0:
        q_replay[:, :h] = inp["k"][:, :h]
    o_replay_full = kernel_call(
        wy_mtp, inp, slice(0, n), inp["S0"], q_override=q_replay
    )
    o_replay = o_replay_full[:, h:]

    # two-call slow path with the same kernel, and the production branch kernel
    o_2call = kernel_call(wy_mtp, inp, slice(h, n), S1_kernel)
    o_branch = kernel_call(branch_mtp, inp, slice(h, n), S1_kernel)

    row = dict(
        e_seq16=maxdiff(o16, o32),
        e_2call=maxdiff(o_2call, o32),
        e_branch=maxdiff(o_branch, o32),
        e_replay=maxdiff(o_replay, o32),
        replay_vs_2call=maxdiff(o_replay, o_2call),
        nan_draft=int(torch.isnan(o_replay).sum().item())
        + int(torch.isinf(o_replay).sum().item()),
        bitexact_h0=(h == 0 and torch.equal(o_replay, o_2call)),
    )
    return row


def nan_probe(B=16, regime="strong-decay", T=4, h=8, seed=7):
    """What happens with ZERO dummy q rows (the l2norm 0/0 case)?"""
    inp = gen_inputs(B, regime, seed)
    n = h + T
    q_zero = inp["q"].clone()
    q_zero[:, :h] = 0.0
    out = kernel_call(wy_mtp, inp, slice(0, n), inp["S0"], q_override=q_zero)
    hist_bad = int(torch.isnan(out[:, :h]).sum() + torch.isinf(out[:, :h]).sum())
    draft_bad = int(torch.isnan(out[:, h:]).sum() + torch.isinf(out[:, h:]).sum())
    print(
        f"\nNaN probe (zeroed hist q, B={B}, T={T}, h={h}, {regime}): "
        f"hist rows nan/inf={hist_bad} (discarded anyway), "
        f"draft rows nan/inf={draft_bad}  {'<-- LEAKS!' if draft_bad else '(no leak)'}"
    )


def packing_cost(Bs=(1, 16, 256), T=4, h=12, iters=200):
    """Per-verify ring packing: copy T draft rows into rows [h:h+T] of persistent
    16-row q/k/v/a/b buffers (hist rows + q dummies already resident)."""
    print("\nPacking cost (copy T draft rows into persistent 16-row buffers):")
    for B in Bs:
        with torch.device("cuda"):
            ring_q = torch.zeros(B, N_MAX, H, K_DIM, dtype=torch.bfloat16)
            ring_k = torch.zeros(B, N_MAX, H, K_DIM, dtype=torch.bfloat16)
            ring_v = torch.zeros(B, N_MAX, HV, V_DIM, dtype=torch.bfloat16)
            ring_a = torch.zeros(B, N_MAX, HV, dtype=torch.bfloat16)
            ring_b = torch.zeros(B, N_MAX, HV, dtype=torch.bfloat16)
            q = torch.randn(B, T, H, K_DIM, dtype=torch.bfloat16)
            k = torch.randn(B, T, H, K_DIM, dtype=torch.bfloat16)
            v = torch.randn(B, T, HV, V_DIM, dtype=torch.bfloat16)
            a = torch.randn(B, T, HV, dtype=torch.bfloat16)
            b = torch.randn(B, T, HV, dtype=torch.bfloat16)
        sl = slice(h, h + T)

        def pack():
            ring_q[:, sl].copy_(q)
            ring_k[:, sl].copy_(k)
            ring_v[:, sl].copy_(v)
            ring_a[:, sl].copy_(a)
            ring_b[:, sl].copy_(b)

        for _ in range(20):
            pack()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)

        def timeit(f):
            times = []
            for _ in range(iters):
                start.record()
                f()
                stop.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(stop) * 1e3)  # us
            times.sort()
            return times[len(times) // 2]

        eager_us = timeit(pack)
        # Serving runs the decode step inside a CUDA graph; the 5 launch gaps
        # above vanish there. Graph-replay is the honest per-verify cost.
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            pack()
        graph_us = timeit(g.replay)
        print(
            f"  B={B:>4}  T={T} rows -> eager {eager_us:8.2f} us | "
            f"graphed {graph_us:8.2f} us (median)"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="B=1, one regime (smoke)")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 256])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    regimes = ["test-mild"] if args.quick else list(REGIMES)
    batches = [1] if args.quick else args.batches

    print(f"GPU: {torch.cuda.get_device_name()}  H={H} HV={HV} K=V={K_DIM}")
    hdr = (
        f"{'regime':>13} {'B':>4} {'T':>3} {'h':>3} | {'seq16':>9} {'2call':>9} "
        f"{'branch':>9} {'REPLAY':>9} | {'rp-vs-2c':>9} {'nan':>4}"
    )
    fails = 0
    for regime in regimes:
        for B in batches:
            inp = gen_inputs(B, regime, args.seed)
            print(f"\n[{regime} B={B}] torch reference sweeps (fp32 + bf16)...", flush=True)
            _, outs32 = torch_sweep(inp, torch.float32)
            states16, outs16 = torch_sweep(inp, torch.bfloat16)
            print(hdr)
            for T, h in COMBOS:
                r = run_config(inp, states16, outs32, outs16, T, h)
                # PASS: replay error at the same scale as the existing paths' error
                # (2x headroom), and nothing non-finite in the draft rows.
                bar = 2.0 * max(r["e_2call"], r["e_branch"], r["e_seq16"])
                ok = r["e_replay"] <= bar and r["nan_draft"] == 0
                fails += 0 if ok else 1
                tag = "" if ok else "  <-- FAIL"
                bit = " (bit==2call)" if r["bitexact_h0"] else ""
                print(
                    f"{regime:>13} {B:>4} {T:>3} {h:>3} | {r['e_seq16']:9.2e} "
                    f"{r['e_2call']:9.2e} {r['e_branch']:9.2e} {r['e_replay']:9.2e} | "
                    f"{r['replay_vs_2call']:9.2e} {r['nan_draft']:>4}{bit}{tag}",
                    flush=True,
                )

    nan_probe()
    packing_cost()
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILING CONFIG(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
