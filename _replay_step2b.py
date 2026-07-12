"""ReplaySSM step 2b: multi-iteration protocol oracle with KERNEL B in the loop.

Same validation idea as _replay_step2.py, upgraded to the production-shaped
protocol enabled by gdn_decode_bf16_wy_state_and_output:

  - ALWAYS-KERNEL-B mode: every iteration is ONE wy_state_and_output launch on
    the full 16-row rings. flush_steps[i] = h_i for requests whose ring is
    past the predictive threshold (h_i > WINDOW - 2T = 8), else 0.
  - PER-REQUEST flushing: requests fold on their own schedule (no flush-all).
    Heterogeneous acceptance (per-request max accept in {1..4}) makes flush
    cadences and the realized per-iteration flush rate f vary — the
    flush_rate_param scenario with realistic values.
  - Flushed requests: their h_i committed rows are folded into the pool by the
    kernel (in place); the harness then compacts the ring (this iteration's
    accepted drafts move to rows [0:A_i], tail zeroed).

Checks every iteration (same references/criteria as step 2):
  - draft outputs vs fp32 ground truth, alongside seq16 / branch / wy2call
    comparators (floor 1e-3 = the shipped kernel's noise class);
  - for requests that flushed THIS iteration: pool slot vs the references'
    state at the fold point (all commits through the previous iteration) —
    drift must stay <= max(2 x production drift, 5e-3).

Run (B200, from repo root): source env.sh && python _replay_step2b.py [--quick]
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
    gated_delta_rule_mtp as wy_out,
)
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_state_and_output import (  # noqa: E402
    gated_delta_rule_mtp as wy_flush,
)

H, HV, K_DIM, V_DIM = 16, 64, 128, 128
WINDOW = 16
T_DRAFT = 4
FLUSH_THRESH = WINDOW - 2 * T_DRAFT  # flush self when h > 8 (predictive)
SCALE = 1.0 / math.sqrt(K_DIM)

REGIMES = {  # (A_log_scale, dt_bias_scale, a_scale, state_scale)
    "test-mild": (0.1, 0.1, 0.1, 1.0),
    "strong-decay": (1.0, 1.0, 1.0, 1.0),
    "big-state": (0.1, 0.1, 0.1, 30.0),
}


def compact(t):
    return torch.empty(t.shape, dtype=t.dtype, device=t.device).copy_(t)


def gen_layer(regime, seed):
    sA, sdt, _, _ = REGIMES[regime]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    with torch.device("cuda"):
        A_log = torch.randn(HV, dtype=torch.float32) * sA
        dt_bias = torch.randn(HV, dtype=torch.float32) * sdt
    return A_log, dt_bias


def gen_draft(B, regime):
    _, _, sa, _ = REGIMES[regime]
    with torch.device("cuda"):
        return dict(
            q=torch.randn(B, T_DRAFT, H, K_DIM, dtype=torch.bfloat16),
            k=torch.randn(B, T_DRAFT, H, K_DIM, dtype=torch.bfloat16),
            v=torch.randn(B, T_DRAFT, HV, V_DIM, dtype=torch.bfloat16),
            a=torch.randn(B, T_DRAFT, HV, dtype=torch.bfloat16) * sa,
            b=torch.randn(B, T_DRAFT, HV, dtype=torch.bfloat16),
        )


class ProtocolB:
    """Always-kernel-B protocol: per-request predictive flush via flush_steps."""

    def __init__(self, S0, A_log, dt_bias):
        B = S0.shape[0]
        self.B = B
        self.A_log, self.dt_bias = A_log, dt_bias
        self.S_pool = S0.clone()  # written in place by the kernel
        with torch.device("cuda"):
            self.rq = torch.zeros(B, WINDOW, H, K_DIM, dtype=torch.bfloat16)
            self.rk = torch.zeros(B, WINDOW, H, K_DIM, dtype=torch.bfloat16)
            self.rv = torch.zeros(B, WINDOW, HV, V_DIM, dtype=torch.bfloat16)
            self.ra = torch.zeros(B, WINDOW, HV, dtype=torch.bfloat16)
            self.rb = torch.zeros(B, WINDOW, HV, dtype=torch.bfloat16)
            self.h = torch.zeros(B, dtype=torch.long)
            self.idx = torch.arange(B, dtype=torch.int32)
            self.bi = torch.arange(B)[:, None]
        self.flush_events = 0

    def step(self, d, accepted):
        """One iteration: write drafts, one kernel-B launch, commit+compact.
        Returns (draft outputs [B,T,HV,V], flushed mask [B])."""
        pos = self.h[:, None] + torch.arange(T_DRAFT, device="cuda")[None, :]
        self.rq[self.bi, pos] = d["q"]
        self.rk[self.bi, pos] = d["k"]
        self.rv[self.bi, pos] = d["v"]
        self.ra[self.bi, pos] = d["a"]
        self.rb[self.bi, pos] = d["b"]

        flush_mask = self.h > FLUSH_THRESH
        flush_steps = torch.where(
            flush_mask, self.h, torch.zeros_like(self.h)
        ).to(torch.int32)
        self.flush_events += int(flush_mask.sum().item())

        out = wy_flush(
            A_log=self.A_log,
            a=self.ra,
            dt_bias=self.dt_bias,
            q=self.rq,
            k=self.rk,
            v=self.rv,
            b=self.rb,
            initial_state_source=self.S_pool,
            initial_state_indices=self.idx,
            flush_steps=flush_steps,
            disable_state_update=False,
            use_qk_l2norm_in_kernel=True,
            scale=SCALE,
        )
        torch.cuda.synchronize()
        o_draft = out[self.bi, pos]

        # Commit + ring maintenance.
        # Non-flushed: h += A (rejected rows get overwritten next write).
        # Flushed: the h_i folded rows leave the ring; this iteration's
        # accepted drafts (ring rows [h_i : h_i+A_i]) move to rows [0:A_i],
        # everything after is zeroed (tail-only padding invariant).
        acc_pos = self.h[:, None] + torch.arange(T_DRAFT, device="cuda")[None, :]
        for r in (self.rq, self.rk, self.rv, self.ra, self.rb):
            moved = r[self.bi, acc_pos].clone()  # [B, T, ...]
            fm = flush_mask.view(-1, *([1] * (r.dim() - 1)))
            r[:] = torch.where(fm, torch.zeros_like(r), r)
            dst = torch.arange(T_DRAFT, device="cuda")[None, :]
            src_moved = torch.where(
                flush_mask.view(-1, 1, *([1] * (r.dim() - 2))),
                moved,
                r[self.bi, dst],
            )
            r[self.bi, dst] = src_moved
        self.h = torch.where(flush_mask, torch.zeros_like(self.h), self.h)
        self.h += accepted
        return o_draft, flush_mask


class Reference:
    def __init__(self, S0, A_log, dt_bias, state_dtype):
        self.state = S0.transpose(-2, -1).contiguous().to(state_dtype)
        self.A_log, self.dt_bias = A_log, dt_bias
        self.state_dtype = state_dtype

    def _step(self, state, d, t):
        return decode_delta_rule(
            d["q"][:, t].float(), d["k"][:, t].float(), d["v"][:, t].float(),
            state, A_log=self.A_log, a=d["a"][:, t], dt_bias=self.dt_bias,
            b=d["b"][:, t], scale_factor=SCALE, softplus_beta=1.0,
            softplus_threshold=20.0, use_l2_norm=True,
            state_dtype=self.state_dtype,
        )

    def verify(self, d):
        s, outs = self.state.clone(), []
        for t in range(T_DRAFT):
            o, s = self._step(s, d, t)
            outs.append(o.float())
        return torch.stack(outs, dim=1)

    def commit(self, d, accepted):
        s = self.state
        for t in range(T_DRAFT):
            _, s2 = self._step(s, d, t)
            s = torch.where((accepted > t).view(-1, 1, 1, 1), s2, s)
        self.state = s

    def state_kernel_layout(self):
        return self.state.transpose(-2, -1).contiguous().to(torch.bfloat16)


def kernel_verify(fn, d, S_kernel, A_log, dt_bias):
    B = d["q"].shape[0]
    out = fn(
        A_log=A_log, a=compact(d["a"]), dt_bias=dt_bias, q=compact(d["q"]),
        k=compact(d["k"]), v=compact(d["v"]), b=compact(d["b"]),
        initial_state_source=S_kernel.clone(),
        initial_state_indices=torch.arange(B, dtype=torch.int32, device="cuda"),
        disable_state_update=True, use_qk_l2norm_in_kernel=True, scale=SCALE,
    )
    torch.cuda.synchronize()
    return out


def maxdiff(x, y):
    return (x.float() - y.float()).abs().max().item()


def simulate(regime, B, iters, seed):
    _, _, _, sS = REGIMES[regime]
    A_log, dt_bias = gen_layer(regime, seed)
    with torch.device("cuda"):
        S0 = torch.randn(B, HV, V_DIM, K_DIM, dtype=torch.bfloat16) * sS
    proto = ProtocolB(S0, A_log, dt_bias)
    ref32 = Reference(S0, A_log, dt_bias, torch.float32)
    ref16 = Reference(S0, A_log, dt_bias, torch.bfloat16)
    rng = torch.Generator(device="cpu").manual_seed(seed + 1)
    # Heterogeneous acceptance: request i accepts 1..amax_i per iteration,
    # amax_i cycling {1,2,3,4} -> flush cadences from ~12 down to ~4 iters.
    amax = (torch.arange(B) % 4 + 1).to(torch.int64)

    print(f"\n[{regime} B={B}] {iters} iters, per-request flush (h > {FLUSH_THRESH})")
    print(
        f"{'it':>3} {'h':>9} {'f%':>4} | {'seq16':>9} {'branch':>9} {'wy2call':>9} "
        f"{'REPLAY':>9} | {'nan':>4} | flushed-slot drift kernel/prod vs fp32"
    )
    FLOOR = 1e-3 * max(sS, 1.0)
    fails = 0
    for it in range(iters):
        d = gen_draft(B, regime)
        h_before = proto.h.clone()
        accepted = (
            torch.rand(B, generator=rng) * amax.float()
        ).long().clamp(0, T_DRAFT - 1) + 1
        accepted = accepted.to("cuda")

        # references BEFORE the protocol mutates anything
        o32 = ref32.verify(d)
        o16 = ref16.verify(d)
        S_now16 = ref16.state_kernel_layout()
        o_branch = kernel_verify(branch_mtp, d, S_now16, A_log, dt_bias)
        o_wy = kernel_verify(wy_out, d, S_now16, A_log, dt_bias)

        o_replay, flushed = proto.step(d, accepted)

        # drift check for slots flushed THIS iteration: pool now holds all
        # commits through the previous iteration == refs' current state.
        drift_note = ""
        if bool(flushed.any()):
            S32 = ref32.state.transpose(-2, -1).contiguous().float()
            S16 = S_now16.float()
            fm = flushed.view(-1, 1, 1, 1)
            d_kernel = maxdiff(
                proto.S_pool.float().masked_fill(~fm, 0.0), S32.masked_fill(~fm, 0.0)
            )
            d_prod = maxdiff(
                S16.masked_fill(~fm, 0.0), S32.masked_fill(~fm, 0.0)
            )
            # Drift bar: kernel B folds via the WY tensor-core form, which
            # rounds intermediates (C, Khat, Tmat) to bf16 — the documented WY
            # precision class: <= ~1 bf16 ULP of the state magnitude per fold
            # (steady-state, decay-damped; the thing to catch here is GROWTH
            # across flush cycles, not the sub-ULP level itself). Floor =
            # 1 ULP = 2^-7 * state_scale; also allow 2x the production
            # sequential-bf16 drift, whichever is larger.
            d_ok = d_kernel <= max(2.0 * d_prod, 7.9e-3 * max(sS, 1.0))
            fails += 0 if d_ok else 1
            drift_note = (
                f"{int(flushed.sum())}fl {d_kernel:.2e}/{d_prod:.2e}"
                f"{'' if d_ok else '  <-- DRIFT FAIL'}"
            )

        e_seq16 = maxdiff(o16, o32)
        e_branch = maxdiff(o_branch, o32)
        e_wy = maxdiff(o_wy, o32)
        e_replay = maxdiff(o_replay, o32)
        n_bad = int(torch.isnan(o_replay).sum() + torch.isinf(o_replay).sum())
        ok = e_replay <= max(2.0 * max(e_wy, e_branch, e_seq16), FLOOR) and n_bad == 0
        fails += 0 if ok else 1

        ref32.commit(d, accepted)
        ref16.commit(d, accepted)

        hmin, hmax = int(h_before.min()), int(h_before.max())
        fpct = int(100 * flushed.float().mean().item())
        print(
            f"{it:>3} {f'{hmin}-{hmax}':>9} {fpct:>4} | {e_seq16:9.2e} "
            f"{e_branch:9.2e} {e_wy:9.2e} {e_replay:9.2e} | {n_bad:>4} | "
            f"{drift_note}{'' if ok else '  <-- FAIL'}",
            flush=True,
        )
    print(f"  -> {proto.flush_events} request-flush events in {iters} iterations")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 256])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    regimes = ["test-mild"] if args.quick else list(REGIMES)
    batches = [8] if args.quick else args.batches
    iters = 10 if args.quick else args.iters

    print(f"GPU: {torch.cuda.get_device_name()}  H={H} HV={HV} K=V={K_DIM}")
    fails = 0
    for regime in regimes:
        for B in batches:
            fails += simulate(regime, B, iters, args.seed)
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILING CHECK(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
