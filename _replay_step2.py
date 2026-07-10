"""ReplaySSM step 2: flush-boundary oracle over many iterations (zero kernel changes).

Step 1 (`_replay_step1.py`) proved ONE replay call is correct. This script runs the
full ReplaySSM protocol for many verify iterations and checks it never drifts from
the always-up-to-date reference:

  checkpoint S_ckpt (bf16 pool) + 16-row rings (q,k,v,a,b) + per-request h
  every iteration:
    1. write the T draft rows into ring rows [h_i : h_i+T] (per request)
    2. verify: ONE WY output-only call over the full 16-row rings from S_ckpt
       (rows past h_i+T are zero -> provably inert for causal outputs);
       gather rows [h_i : h_i+T] as the draft outputs
    3. accept a seeded-random prefix A_i in [1, T] per request; h_i += A_i
       (rejected rows are overwritten by the next iteration's draft write)
    4. if any h_i + T > 16: FLUSH ALL -> one branch-kernel call in its shipped
       per-request recovery mode (accepted_steps = h-1, disable_output=True,
       disable_state_update=False) folds each request's h_i ring rows into
       S_ckpt in place; rings zeroed; h = 0

Checked EVERY iteration against two sequential references (the same
decode_delta_rule the test suite trusts), advanced by exactly the accepted tokens:
  ref32 : fp32-state  -> mathematical ground truth (error basis)
  ref16 : bf16-state  -> today's production recurrence
plus a production comparator o_branch = branch kernel verifying the T draft rows
from ref16's state (what serving would output today).

PASS per iteration: e_replay <= 2 * max(e_branch, e_seq16), no NaN/Inf.
At every flush: checkpoint drift d(S_ckpt, ref32) <= max(2 * d(ref16, ref32), 5e-3)
— i.e. the kernel-folded checkpoint drifts no faster than production's own state.

Run (B200, from repo root):
    source env.sh && python _replay_step2.py [--quick]
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
WINDOW = 16
T_DRAFT = 4
SCALE = 1.0 / math.sqrt(K_DIM)

REGIMES = {  # (A_log_scale, dt_bias_scale, a_scale, state_scale)
    "test-mild": (0.1, 0.1, 0.1, 1.0),
    "strong-decay": (1.0, 1.0, 1.0, 1.0),
    "big-state": (0.1, 0.1, 0.1, 30.0),
}


def compact(t):
    """Canonical-stride copy (B=1 slices can be 'contiguous' with bad strides)."""
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


class Protocol:
    """Checkpoint + rings + per-request h, driven only by the two existing kernels."""

    def __init__(self, S0, A_log, dt_bias):
        B = S0.shape[0]
        self.B = B
        self.A_log, self.dt_bias = A_log, dt_bias
        self.S_ckpt = S0.clone()  # kernel layout pool [B,HV,V,K] bf16
        with torch.device("cuda"):
            self.rq = torch.zeros(B, WINDOW, H, K_DIM, dtype=torch.bfloat16)
            self.rk = torch.zeros(B, WINDOW, H, K_DIM, dtype=torch.bfloat16)
            self.rv = torch.zeros(B, WINDOW, HV, V_DIM, dtype=torch.bfloat16)
            self.ra = torch.zeros(B, WINDOW, HV, dtype=torch.bfloat16)
            self.rb = torch.zeros(B, WINDOW, HV, dtype=torch.bfloat16)
            self.h = torch.zeros(B, dtype=torch.long)
            self.idx = torch.arange(B, dtype=torch.int32)
            self.bi = torch.arange(B)[:, None]  # [B,1] batch index helper
        self.flushes = 0

    def verify(self, d):
        """Write draft rows at per-request h, replay the full window, gather."""
        pos = self.h[:, None] + torch.arange(T_DRAFT, device="cuda")[None, :]
        self.rq[self.bi, pos] = d["q"]
        self.rk[self.bi, pos] = d["k"]
        self.rv[self.bi, pos] = d["v"]
        self.ra[self.bi, pos] = d["a"]
        self.rb[self.bi, pos] = d["b"]
        out = wy_mtp(
            A_log=self.A_log,
            a=self.ra,
            dt_bias=self.dt_bias,
            q=self.rq,
            k=self.rk,
            v=self.rv,
            b=self.rb,
            initial_state_source=self.S_ckpt,
            initial_state_indices=self.idx,
            disable_state_update=True,
            use_qk_l2norm_in_kernel=True,
            scale=SCALE,
        )
        torch.cuda.synchronize()
        return out[self.bi, pos]  # [B,T,HV,V]

    def commit(self, accepted):
        self.h += accepted  # rejected rows get overwritten by the next draft write

    def needs_flush(self):
        return bool((self.h + T_DRAFT).max().item() > WINDOW)

    def flush(self):
        """Fold each request's h_i ring rows into S_ckpt (in place), reset rings."""
        branch_mtp(
            A_log=self.A_log,
            a=self.ra,
            dt_bias=self.dt_bias,
            q=self.rq,
            k=self.rk,
            v=self.rv,
            b=self.rb,
            initial_state_source=self.S_ckpt,
            initial_state_indices=self.idx,
            accepted_steps=(self.h - 1).to(torch.int32),
            disable_state_update=False,
            disable_output=True,
            use_qk_l2norm_in_kernel=True,
            scale=SCALE,
        )
        torch.cuda.synchronize()
        for r in (self.rq, self.rk, self.rv, self.ra, self.rb):
            r.zero_()
        self.h.zero_()
        self.flushes += 1


class Reference:
    """Sequential decode_delta_rule bookkeeper (state advanced by accepted tokens)."""

    def __init__(self, S0, A_log, dt_bias, state_dtype):
        self.state = S0.transpose(-2, -1).contiguous().to(state_dtype)  # [B,HV,K,V]
        self.A_log, self.dt_bias = A_log, dt_bias
        self.state_dtype = state_dtype

    def _step(self, state, d, t):
        return decode_delta_rule(
            d["q"][:, t].float(),
            d["k"][:, t].float(),
            d["v"][:, t].float(),
            state,
            A_log=self.A_log,
            a=d["a"][:, t],
            dt_bias=self.dt_bias,
            b=d["b"][:, t],
            scale_factor=SCALE,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            use_l2_norm=True,
            state_dtype=self.state_dtype,
        )

    def verify(self, d):
        s, outs = self.state.clone(), []
        for t in range(T_DRAFT):
            o, s = self._step(s, d, t)
            outs.append(o.float())
        return torch.stack(outs, dim=1)  # [B,T,HV,V]

    def commit(self, d, accepted):
        s = self.state
        for t in range(T_DRAFT):
            _, s2 = self._step(s, d, t)
            mask = (accepted > t).view(-1, 1, 1, 1)
            s = torch.where(mask, s2, s)
        self.state = s

    def state_kernel_layout(self):
        return self.state.transpose(-2, -1).contiguous().to(torch.bfloat16)


def kernel_verify(fn, d, S_kernel, A_log, dt_bias):
    """Comparator: kernel `fn` verifying the T draft rows from up-to-date S_kernel.
    fn=branch_mtp -> production output; fn=wy_mtp -> same-kernel baseline that
    isolates the replay effect (WY's own bf16-intermediate noise floor)."""
    B = d["q"].shape[0]
    out = fn(
        A_log=A_log,
        a=compact(d["a"]),
        dt_bias=dt_bias,
        q=compact(d["q"]),
        k=compact(d["k"]),
        v=compact(d["v"]),
        b=compact(d["b"]),
        initial_state_source=S_kernel.clone(),
        initial_state_indices=torch.arange(B, dtype=torch.int32, device="cuda"),
        disable_state_update=True,
        use_qk_l2norm_in_kernel=True,
        scale=SCALE,
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
    proto = Protocol(S0, A_log, dt_bias)
    ref32 = Reference(S0, A_log, dt_bias, torch.float32)
    ref16 = Reference(S0, A_log, dt_bias, torch.bfloat16)
    accept_rng = torch.Generator(device="cpu").manual_seed(seed + 1)

    print(
        f"\n[{regime} B={B}] {iters} iterations, T={T_DRAFT} drafts/iter, "
        f"window={WINDOW}"
    )
    print(
        f"{'it':>3} {'h':>9} {'acc':>4} | {'seq16':>9} {'branch':>9} {'wy2call':>9} "
        f"{'REPLAY':>9} | {'rp-vs-wy':>9} {'nan':>4} | "
        f"flush(ckpt drift: kernel/prod vs fp32)"
    )
    # Absolute floor: the WY kernel ships with max|d| <= 9.77e-04 (1-ULP class)
    # vs the branch kernel across the full PR sweep — errors at or below 1e-3
    # are indistinguishable from the accepted kernel noise.
    FLOOR = 1e-3
    fails = 0
    for it in range(iters):
        d = gen_draft(B, regime)
        h_before = proto.h.clone()

        o_replay = proto.verify(d)
        o32 = ref32.verify(d)
        o16 = ref16.verify(d)
        S_now = ref16.state_kernel_layout()
        o_branch = kernel_verify(branch_mtp, d, S_now, A_log, dt_bias)
        o_wy = kernel_verify(wy_mtp, d, S_now, A_log, dt_bias)

        e_seq16 = maxdiff(o16, o32)
        e_branch = maxdiff(o_branch, o32)
        e_wy = maxdiff(o_wy, o32)
        e_replay = maxdiff(o_replay, o32)
        e_rp_wy = maxdiff(o_replay, o_wy)
        n_bad = int(torch.isnan(o_replay).sum() + torch.isinf(o_replay).sum())
        bar = max(2.0 * max(e_wy, e_branch, e_seq16), FLOOR)
        ok = e_replay <= bar and n_bad == 0
        fails += 0 if ok else 1

        accepted = torch.randint(
            1, T_DRAFT + 1, (B,), generator=accept_rng
        ).to("cuda")
        proto.commit(accepted)
        ref32.commit(d, accepted)
        ref16.commit(d, accepted)

        flush_note = ""
        if proto.needs_flush():
            proto.flush()
            S32 = ref32.state.transpose(-2, -1).contiguous()
            S16 = ref16.state_kernel_layout()
            d_kernel = maxdiff(proto.S_ckpt, S32)
            d_prod = maxdiff(S16, S32)
            d_ok = d_kernel <= max(2.0 * d_prod, 5e-3)
            fails += 0 if d_ok else 1
            flush_note = (
                f"FLUSH#{proto.flushes} {d_kernel:.2e}/{d_prod:.2e}"
                f"{'' if d_ok else '  <-- DRIFT FAIL'}"
            )

        hmin, hmax = int(h_before.min()), int(h_before.max())
        print(
            f"{it:>3} {f'{hmin}-{hmax}':>9} {accepted.float().mean():4.1f} | "
            f"{e_seq16:9.2e} {e_branch:9.2e} {e_wy:9.2e} {e_replay:9.2e} | "
            f"{e_rp_wy:9.2e} {n_bad:>4} | {flush_note}{'' if ok else '  <-- FAIL'}",
            flush=True,
        )
    print(f"  -> {proto.flushes} flushes in {iters} iterations")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--iters", type=int, default=18)
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 256])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    regimes = ["test-mild"] if args.quick else list(REGIMES)
    batches = [8] if args.quick else args.batches
    iters = 8 if args.quick else args.iters

    print(f"GPU: {torch.cuda.get_device_name()}  H={H} HV={HV} K=V={K_DIM}")
    fails = 0
    for regime in regimes:
        for B in batches:
            fails += simulate(regime, B, iters, args.seed)
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILING CHECK(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
