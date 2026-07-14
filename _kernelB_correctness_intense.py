"""Intense correctness campaign for kernel B (gdn_decode_bf16_wy_state_and_output),
BOTH variants: fused (outputs + fold) and state-only (disable_output=True).

Oracles (the RECURRENT kernel gdn_decode_bf16_state.gated_delta_rule_mtp is the one
independent implementation, used for both sides):
  outputs: wy_output_only (BIT, same-algebra regression), recurrent output mode
           (<= tolerance, all B), torch decode_delta_rule fp32/bf16 (B<=64, seed 0)
  state:   recurrent state-only fold (accepted_steps, all B, masked to P>0),
           torch refs (B<=64, seed 0); floor = 1 bf16 ULP x state_scale
  cross:   state-only fold BIT-equal to fused fold (same math path)

Sections:
  A. shape/seed/P sweep         B. P edge cases incl. eviction folds P in {13..16}
  C. isolation & shuffled pool  D. adversarial values (decay/state/NaN-drafts/q-NaN)
  E. non-contiguous inputs      F. allocator churn (_BF16_CACHE / _DUMMY_OUT class)
  G. CUDA-graph replay vs eager (mutating flush_steps in place)
  H. mixed-config interleaving  I. 200-fold in-place chain soak vs fp32 shadow
  J. concurrent-stream overlap (state-only side stream + wy_output_only verify)

Run (B200): source env.sh && python _kernelB_correctness_intense.py
"""

import math
import os
import sys

import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "tests", "gdn"))
from reference_delta_rule import decode_delta_rule  # noqa: E402

from flashinfer.gdn_kernels.gdn_decode_bf16_state import (  # noqa: E402
    gated_delta_rule_mtp as recurrent,
)
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_output_only import (  # noqa: E402
    gated_delta_rule_mtp as wy_out,
)
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_state_and_output import (  # noqa: E402
    gated_delta_rule_mtp as wy_flush,
)

K_DIM = V_DIM = 128
T = 16
SCALE = 1.0 / math.sqrt(K_DIM)
DEV = "cuda"
FAILURES: list = []
ULP = 2.0 ** -7  # bf16 eps at |1|

HEAD_CONFIGS = [(16, 64), (16, 32), (4, 16), (8, 32)]  # (H=HK, HV); (4,16) = TP4


def fail(tag, msg):
    print(f"FAIL {tag}: {msg}", flush=True)
    FAILURES.append(tag)


def mk(B, HV, H, seed, pool=None, state_scale=1.0, a_scale=0.1, extreme=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    pool = pool or B

    def rn(*shape, scale=1.0):
        return (
            torch.randn(*shape, generator=g, dtype=torch.bfloat16, device=DEV) * scale
        )

    i = dict(
        q=rn(B, T, H, K_DIM),
        k=rn(B, T, H, K_DIM),
        v=rn(B, T, HV, V_DIM),
        a=rn(B, T, HV, scale=a_scale),
        b=rn(B, T, HV),
        A_log=torch.randn(HV, generator=g, dtype=torch.float32, device=DEV) * 0.1,
        dt_bias=torch.randn(HV, generator=g, dtype=torch.float32, device=DEV) * 0.1,
        state=torch.randn(
            pool, HV, V_DIM, K_DIM, generator=g, dtype=torch.bfloat16, device=DEV
        )
        * state_scale,
        idx=torch.arange(B, dtype=torch.int32, device=DEV),
        HV=HV,
        H=H,
        B=B,
        state_scale=state_scale,
    )
    if extreme:
        i["A_log"] = torch.linspace(-8.0, 4.0, HV, device=DEV)
        i["dt_bias"] = torch.linspace(-6.0, 6.0, HV, device=DEV)
    return i


def common(i):
    return dict(
        A_log=i["A_log"], a=i["a"], dt_bias=i["dt_bias"], q=i["q"], k=i["k"],
        v=i["v"], b=i["b"], initial_state_indices=i["idx"],
        use_qk_l2norm_in_kernel=True, scale=SCALE,
    )


def run_fused(i, P):
    S = i["state"].clone()
    o = wy_flush(**common(i), initial_state_source=S, flush_steps=P,
                 disable_state_update=False)
    torch.cuda.synchronize()
    return o, S


def run_stateonly(i, P):
    S = i["state"].clone()
    r = wy_flush(**common(i), initial_state_source=S, flush_steps=P,
                 disable_state_update=False, disable_output=True)
    torch.cuda.synchronize()
    assert r is None
    return S


def run_wyout(i):
    o = wy_out(**common(i), initial_state_source=i["state"].clone(),
               disable_state_update=True)
    torch.cuda.synchronize()
    return o


def run_recurrent_out(i):
    o = recurrent(**common(i), initial_state_source=i["state"].clone(),
                  disable_state_update=True)
    torch.cuda.synchronize()
    return o


def run_recurrent_fold(i, P):
    S = i["state"].clone()
    Pb = torch.where(P > 0, P, torch.ones_like(P))
    recurrent(**common(i), initial_state_source=S,
              accepted_steps=(Pb - 1).to(torch.int32),
              disable_state_update=False, disable_output=True)
    torch.cuda.synchronize()
    return S  # only P>0 slots meaningful


def torch_refs(i, P):
    """Sequential reference: outputs (all 16 rows, fp32+bf16 state) and the
    per-request folded state at P (both dtypes). Assumes idx == arange."""
    B = i["B"]
    res = {}
    for sd in (torch.float32, torch.bfloat16):
        st = i["state"][:B].transpose(-2, -1).contiguous().to(sd)  # [B,HV,K,V]
        states = [st.clone()]
        outs = []
        for t in range(T):
            o, st = decode_delta_rule(
                i["q"][:, t].float(), i["k"][:, t].float(), i["v"][:, t].float(),
                st, A_log=i["A_log"], a=i["a"][:, t], dt_bias=i["dt_bias"],
                b=i["b"][:, t], scale_factor=SCALE, softplus_beta=1.0,
                softplus_threshold=20.0, use_l2_norm=True, state_dtype=sd,
            )
            states.append(st.clone())
            outs.append(o.float())
        stk = torch.stack(states, 0)
        sel = stk[P.long().clamp(min=0), torch.arange(B, device=DEV)]
        key = "32" if sd == torch.float32 else "16"
        res["o" + key] = torch.stack(outs, dim=1)
        res["S" + key] = sel.transpose(-2, -1).contiguous().float()
    return res


def md(x, y, mask=None):
    d = (x.float() - y.float()).abs()
    if mask is not None:
        d = d.masked_fill(~mask, 0.0)
    return d.max().item() if d.numel() else 0.0


def check_config(tag, i, P, do_torch):
    """All oracle checks for one (inputs, P)."""
    ok = True
    o_f, S_f = run_fused(i, P)
    S_so = run_stateonly(i, P)
    o_wy = run_wyout(i)
    o_rec = run_recurrent_out(i)
    S_rec = run_recurrent_fold(i, P)
    B = i["B"]
    sscale = max(i["state_scale"], 1.0)
    mask = (P > 0).view(B, 1, 1, 1)

    if not torch.equal(o_f, o_wy):
        fail(tag, f"fused outputs != wy_output_only (bit), max|d|={md(o_f, o_wy):.2e}")
        ok = False
    if not torch.equal(S_so, S_f):
        fail(tag, f"state-only fold != fused fold (bit), max|d|={md(S_so, S_f):.2e}")
        ok = False
    um = ~(P > 0)
    if bool(um.any()):
        if not torch.equal(S_f[:B][um], i["state"][:B][um]):
            fail(tag, "P=0 slot state modified")
            ok = False
    d_out_rec = md(o_f, o_rec)
    # WY-vs-recurrent output noise scales with |H0| (documented; measured
    # ~2.1-2.5e-3 * scale at scale 30/100); outputs are also BIT-checked vs
    # the shipped wy_output_only above, so this is a cross-impl sanity bar.
    if d_out_rec > 4e-3 * sscale:
        fail(tag, f"fused outputs vs recurrent: {d_out_rec:.2e} > {4e-3 * sscale:.0e}")
        ok = False
    if bool((P > 0).any()) and not do_torch:
        # cross-implementation sanity bar, only when no fp32 truth exists for
        # this config (both folds are bf16-rounded; at extreme scales this
        # delta is dominated by the RECURRENT kernel's own noise)
        d_S_rec = md(S_f[:B], S_rec[:B], mask)
        if d_S_rec > max(8 * ULP * sscale, 2e-2 * sscale):
            fail(tag, f"fold vs recurrent fold: {d_S_rec:.2e}")
            ok = False
    if do_torch:
        r = torch_refs(i, P)
        d_o = md(o_f, r["o32"])
        bar_o = max(2 * md(o_rec, r["o32"]), 2 * md(r["o16"], r["o32"]),
                    2e-3 * sscale)
        if d_o > bar_o:
            fail(tag, f"outputs vs fp32 ref: {d_o:.2e} > bar {bar_o:.2e}")
            ok = False
        if bool((P > 0).any()):
            d_s = md(S_f[:B], r["S32"], mask)
            bar_s = max(
                2 * md(S_rec[:B], r["S32"], mask),
                2 * md(r["S16"], r["S32"], mask),
                2 * ULP * sscale,  # kernel B rounds C and the final store
            )
            if d_s > bar_s:
                fail(tag, f"fold vs fp32 ref: {d_s:.2e} > bar {bar_s:.2e}")
                ok = False
    return ok


def p_patterns(B, seed):
    g = torch.Generator().manual_seed(seed + 1000)
    pats = {
        "uni12": torch.full((B,), 12, dtype=torch.int32, device=DEV),
        "rand": torch.randint(0, 13, (B,), generator=g).to(torch.int32).to(DEV),
        "zero": torch.zeros(B, dtype=torch.int32, device=DEV),
        "one": torch.zeros(B, dtype=torch.int32, device=DEV),
    }
    pats["one"][min(1, B - 1)] = 7
    if B == 1:
        pats["rand"][0] = 9  # avoid vacuous B=1 masks
    return pats


def section_a():
    print("== A. shape/seed/P sweep ==", flush=True)
    for H, HV in HEAD_CONFIGS:
        for B in (1, 2, 3, 8, 17, 64, 177, 256):
            for seed in (0, 1, 2):
                i = mk(B, HV, H, seed)
                for pname, P in p_patterns(B, seed).items():
                    do_torch = B <= 64 and seed == 0
                    check_config(
                        f"A HV={HV} H={H} B={B} s={seed} P={pname}", i, P, do_torch
                    )
        print(f"  config (H={H}, HV={HV}) done", flush=True)
    print("A done", flush=True)


def section_b():
    print("== B. P edge cases (incl. eviction folds) ==", flush=True)
    for H, HV in ((16, 64), (4, 16)):
        i = mk(16, HV, H, 7)
        P = torch.arange(1, 17, dtype=torch.int32, device=DEV)
        check_config(f"B perP HV={HV}", i, P, do_torch=True)
        for pval in (13, 14, 15, 16):
            i = mk(8, HV, H, 100 + pval)
            for t_ in (i["q"], i["k"], i["v"], i["a"], i["b"]):
                t_[:, pval:] = 0
            P = torch.full((8,), pval, dtype=torch.int32, device=DEV)
            check_config(f"B evict P={pval} HV={HV}", i, P, do_torch=True)
    print("B done", flush=True)


def section_c():
    print("== C. isolation & shuffled pool ==", flush=True)
    H, HV, B = 16, 64, 17
    i = mk(B, HV, H, 3, pool=64)
    gcpu = torch.Generator().manual_seed(3)
    i["idx"] = torch.randperm(64, generator=gcpu)[:B].to(torch.int32).to(DEV)
    P = p_patterns(B, 3)["rand"]
    o_f, S_f = run_fused(i, P)
    S_so = run_stateonly(i, P)
    if not torch.equal(S_so, S_f):
        fail("C pool", "state-only != fused with shuffled pool")
    untouched = torch.ones(64, dtype=torch.bool, device=DEV)
    untouched[i["idx"][P > 0].long()] = False
    if not torch.equal(S_f[untouched], i["state"][untouched]):
        fail("C pool", "non-folded pool slots modified")
    i = mk(B, HV, H, 4)
    P = p_patterns(B, 4)["rand"]
    o_bat, S_bat = run_fused(i, P)
    for r in range(B):
        ii = dict(i)
        for kk in ("q", "k", "v", "a", "b", "state"):
            ii[kk] = i[kk][r : r + 1].clone()
        ii.update(B=1, idx=torch.zeros(1, dtype=torch.int32, device=DEV))
        o1, S1 = run_fused(ii, P[r : r + 1].clone())
        if not torch.equal(o1[0], o_bat[r]) or not torch.equal(S1[0], S_bat[r]):
            fail("C iso", f"request {r}: batched != single-request (bit)")
            break
    print("C done", flush=True)


def section_d():
    print("== D. adversarial values ==", flush=True)
    H, HV, B = 16, 64, 8
    P = torch.full((B,), 10, dtype=torch.int32, device=DEV)
    for sscale in (1.0, 30.0, 100.0):
        i = mk(B, HV, H, 11, state_scale=sscale, extreme=True)
        i["b"][:, :, :] = 20.0
        i["b"][:, ::2, :] = -20.0
        check_config(f"D extreme s={sscale:g}", i, P, do_torch=True)
    i = mk(B, HV, H, 12)
    i["k"][:, 2:5] = 0
    i["v"][:, 6:8] = 0
    check_config("D zero-rows", i, P, do_torch=True)
    i = mk(B, HV, H, 13)
    i_clean = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in i.items()}
    for t_ in (i["q"], i["k"], i["v"], i["a"], i["b"]):
        t_[:, 12:14] = float("nan")
        t_[:, 14:16] = float("inf")
    _, S_dirty = run_fused(i, P)
    _, S_clean = run_fused(i_clean, P)
    if not torch.equal(S_dirty, S_clean):
        fail("D nan-drafts", "NaN/Inf in draft rows leaked into the fold")
    i = mk(B, HV, H, 14)
    S_ref = run_stateonly(i, P)
    i["q"][:] = float("nan")
    S_nanq = run_stateonly(i, P)
    if not torch.equal(S_ref, S_nanq):
        fail("D q-nan", "state-only fold depends on q")
    print("D done", flush=True)


def section_e():
    print("== E. non-contiguous / stride edges ==", flush=True)
    H, HV, B = 16, 64, 8
    P = torch.full((B,), 9, dtype=torch.int32, device=DEV)
    i = mk(B, HV, H, 21)
    o_ref, S_ref = run_fused(i, P)
    inc = dict(i)
    for kk in ("q", "k", "v"):
        inc[kk] = i[kk].transpose(1, 2).contiguous().transpose(1, 2)
    o_nc, S_nc = run_fused(inc, P)
    if not torch.equal(o_nc, o_ref) or not torch.equal(S_nc, S_ref):
        fail("E transpose", "non-contig q/k/v changed results")
    for Bs in (1, 3):
        big = mk(Bs + 4, HV, H, 22)
        sl = dict(big)
        for kk in ("q", "k", "v", "a", "b"):
            sl[kk] = big[kk][:Bs]
        sl.update(B=Bs, idx=torch.arange(Bs, dtype=torch.int32, device=DEV),
                  state=big["state"][:Bs].clone())
        cm = dict(sl)
        for kk in ("q", "k", "v", "a", "b"):
            cm[kk] = torch.empty(
                sl[kk].shape, dtype=sl[kk].dtype, device=DEV
            ).copy_(sl[kk])
        Ps = torch.full((Bs,), 8, dtype=torch.int32, device=DEV)
        try:
            o_s, S_s = run_fused(sl, Ps)
            o_c, S_c = run_fused(cm, Ps)
            if not torch.equal(o_s, o_c) or not torch.equal(S_s, S_c):
                fail("E slice", f"B={Bs} sliced-batch results differ from compact")
        except Exception as e:  # loud failure acceptable; silent wrong is not
            print(f"  E slice B={Bs}: raised {type(e).__name__} (acceptable-loud)",
                  flush=True)
    P2 = torch.zeros(2 * B, dtype=torch.int32, device=DEV)
    P2[::2] = 9
    o_nc2, S_nc2 = run_fused(i, P2[::2])
    if not torch.equal(o_nc2, o_ref) or not torch.equal(S_nc2, S_ref):
        fail("E flush_steps", "non-contig flush_steps changed results")
    try:
        bad = mk(B, HV, H, 23, pool=2 * B)
        badd = dict(bad)
        badd["state"] = bad["state"][::2]
        wy_flush(**common(badd), initial_state_source=badd["state"],
                 flush_steps=P, disable_state_update=False)
        fail("E state", "non-contiguous initial_state_source did not raise")
    except AssertionError:
        pass
    print("E done", flush=True)


def section_f():
    print("== F. allocator churn ==", flush=True)
    H, HV, B = 16, 64, 8
    P = torch.full((B,), 11, dtype=torch.int32, device=DEV)
    i = mk(B, HV, H, 31)
    golden_o, golden_S = run_fused(i, P)
    bad = 0
    for it in range(24):
        junk = torch.randn(HV, device=DEV) * 5.0
        junk2 = torch.randn(HV, device=DEV) * 5.0
        del junk, junk2
        i["A_log"] = i["A_log"].clone()  # new storage, same values
        i["dt_bias"] = i["dt_bias"].clone()
        Pc = P.clone()
        o, S = run_fused(i, Pc)
        S2 = run_stateonly(i, Pc)
        if (not torch.equal(o, golden_o) or not torch.equal(S, golden_S)
                or not torch.equal(S2, golden_S)):
            bad += 1
    if bad:
        fail("F churn", f"{bad}/24 iterations diverged (stale-cast class)")
    for HV2, H2 in ((16, 4), (64, 16), (32, 8), (64, 16)):
        ii = mk(4, HV2, H2, 32)
        run_stateonly(ii, torch.full((4,), 6, dtype=torch.int32, device=DEV))
    print("F done", flush=True)


def section_g():
    print("== G. CUDA-graph replay vs eager ==", flush=True)
    H, HV, B = 16, 64, 32
    i = mk(B, HV, H, 41)
    P = p_patterns(B, 41)["rand"]
    S_pool = i["state"].clone()
    out_holder = {}

    def fused_call():
        out_holder["o"] = wy_flush(**common(i), initial_state_source=S_pool,
                                   flush_steps=P, disable_state_update=False)

    def so_call():
        wy_flush(**common(i), initial_state_source=S_pool, flush_steps=P,
                 disable_state_update=False, disable_output=True)

    for name, call in (("fused", fused_call), ("stateonly", so_call)):
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call()
        for rep in range(5):
            i2 = mk(B, HV, H, 42 + rep)
            P2 = p_patterns(B, 42 + rep)["rand"]
            # A_log/dt_bias stay FIXED across replays: they are per-layer
            # constants by contract (_BF16_CACHE identity-caches their bf16
            # casts); the eager comparison below uses the same constants.
            i2["A_log"] = i["A_log"]
            i2["dt_bias"] = i["dt_bias"]
            for kk in ("q", "k", "v", "a", "b"):
                i[kk].copy_(i2[kk])
            P.copy_(P2)
            S_pool.copy_(i2["state"])
            g.replay()
            torch.cuda.synchronize()
            S_graph = S_pool.clone()
            o_graph = out_holder["o"].clone() if name == "fused" else None
            o_eager, S_eager = run_fused(i2, P2)
            if not torch.equal(S_graph, S_eager):
                fail(f"G {name}", f"replay {rep}: state graph != eager")
                break
            if name == "fused" and not torch.equal(o_graph, o_eager):
                fail(f"G {name}", f"replay {rep}: outputs graph != eager")
                break
    print("G done", flush=True)


def section_h():
    print("== H. mixed-config interleaving ==", flush=True)
    cfgs = [(16, 64, "fused"), (4, 16, "so"), (8, 32, "fused"), (16, 64, "so")]
    golden = []
    for H, HV, mode in cfgs:
        i = mk(8, HV, H, 51)
        P = torch.full((8,), 10, dtype=torch.int32, device=DEV)
        if mode == "fused":
            golden.append(run_fused(i, P))
        else:
            golden.append((None, run_stateonly(i, P)))
    for rep in range(3):
        for (H, HV, mode), (go, gS) in zip(cfgs, golden):
            i = mk(8, HV, H, 51)
            P = torch.full((8,), 10, dtype=torch.int32, device=DEV)
            if mode == "fused":
                o, S = run_fused(i, P)
                if not torch.equal(o, go) or not torch.equal(S, gS):
                    fail("H", f"rep {rep} HV={HV} {mode}: stale-cache divergence")
            else:
                S = run_stateonly(i, P)
                if not torch.equal(S, gS):
                    fail("H", f"rep {rep} HV={HV} {mode}: stale-cache divergence")
    print("H done", flush=True)


def section_i():
    print("== I. 200-fold in-place chain soak ==", flush=True)
    H, HV, B = 4, 16, 4
    torch.manual_seed(61)
    S_pool = torch.randn(B, HV, V_DIM, K_DIM, dtype=torch.bfloat16, device=DEV)
    shadow = S_pool.transpose(-2, -1).contiguous().float()  # [B,HV,K,V] fp32
    A_log = torch.randn(HV, device=DEV) * 0.1
    dt_bias = torch.randn(HV, device=DEV) * 0.1
    idx = torch.arange(B, dtype=torch.int32, device=DEV)
    drifts = []
    for step in range(200):
        i = mk(B, HV, H, 1000 + step)
        i.update(A_log=A_log, dt_bias=dt_bias, idx=idx)
        pval = 8 + (step % 9)  # 8..16
        for t_ in (i["q"], i["k"], i["v"], i["a"], i["b"]):
            t_[:, pval:] = 0
        P = torch.full((B,), pval, dtype=torch.int32, device=DEV)
        if step % 2 == 0:
            wy_flush(**common(i), initial_state_source=S_pool, flush_steps=P,
                     disable_state_update=False, disable_output=True)
        else:
            wy_flush(**common(i), initial_state_source=S_pool, flush_steps=P,
                     disable_state_update=False)
        torch.cuda.synchronize()
        st = shadow
        for t in range(pval):
            _, st = decode_delta_rule(
                i["q"][:, t].float(), i["k"][:, t].float(), i["v"][:, t].float(),
                st, A_log=A_log, a=i["a"][:, t], dt_bias=dt_bias, b=i["b"][:, t],
                scale_factor=SCALE, softplus_beta=1.0, softplus_threshold=20.0,
                use_l2_norm=True, state_dtype=torch.float32,
            )
        shadow = st
        drifts.append(
            (S_pool.float() - shadow.transpose(-2, -1)).abs().max().item()
        )
    mx = max(drifts)
    first, last = max(drifts[:50]), max(drifts[-50:])
    print(f"  drift max={mx:.2e} first50={first:.2e} last50={last:.2e}", flush=True)
    if mx > 4 * ULP or last > 2 * first + ULP:
        fail("I soak",
             f"drift unbounded: max={mx:.2e} first={first:.2e} last={last:.2e}")
    print("I done", flush=True)


def section_j():
    print("== J. concurrent-stream overlap ==", flush=True)
    H, HV, B = 16, 64, 64
    half = B // 2
    for rep in range(20):
        i = mk(B, HV, H, 71 + rep)
        P = torch.full((half,), 12, dtype=torch.int32, device=DEV)
        fold_in = dict(i)
        ver_in = dict(i)
        for kk in ("q", "k", "v", "a", "b"):
            fold_in[kk] = i[kk][:half].contiguous()
            ver_in[kk] = i[kk][half:].contiguous()
        fold_in.update(B=half,
                       idx=torch.arange(half, dtype=torch.int32, device=DEV))
        ver_in.update(B=half,
                      idx=torch.arange(half, B, dtype=torch.int32, device=DEV))
        S_seq = i["state"].clone()
        wy_flush(**common(fold_in), initial_state_source=S_seq, flush_steps=P,
                 disable_state_update=False, disable_output=True)
        o_seq = wy_out(**common(ver_in), initial_state_source=S_seq,
                       disable_state_update=True)
        torch.cuda.synchronize()
        S_con = i["state"].clone()
        side = torch.cuda.Stream()
        ev = torch.cuda.Event()
        with torch.cuda.stream(side):
            wy_flush(**common(fold_in), initial_state_source=S_con, flush_steps=P,
                     disable_state_update=False, disable_output=True)
            ev.record(side)
        o_con = wy_out(**common(ver_in), initial_state_source=S_con,
                       disable_state_update=True)
        torch.cuda.current_stream().wait_event(ev)
        torch.cuda.synchronize()
        if not torch.equal(S_con, S_seq) or not torch.equal(o_con, o_seq):
            fail("J overlap", f"rep {rep}: concurrent != sequential")
            break
    print("J done", flush=True)


def section_k():
    print("== K. request_indices: scattered-indirect vs packed ==", flush=True)
    H, HV = 16, 64
    for B_full, n_sub in ((64, 21), (256, 85)):
        i = mk(B_full, HV, H, 81)
        g = torch.Generator().manual_seed(81 + B_full)
        sub = (
            torch.randperm(B_full, generator=g)[:n_sub]
            .sort().values.to(torch.int32).to(DEV)
        )
        P_sub = torch.randint(0, 13, (n_sub,), generator=g).to(torch.int32).to(DEV)
        P_sub[0] = 12
        P_sub[1] = 0
        base = dict(A_log=i["A_log"], dt_bias=i["dt_bias"],
                    use_qk_l2norm_in_kernel=True, scale=SCALE)

        # indirect: FULL interleaved tensors + request_indices (state slots
        # default to the ring rows)
        S_ind = i["state"].clone()
        o_ind = wy_flush(**base, q=i["q"], k=i["k"], v=i["v"], a=i["a"],
                         b=i["b"], initial_state_source=S_ind,
                         request_indices=sub, flush_steps=P_sub,
                         disable_state_update=False)
        S_ind_so = i["state"].clone()
        wy_flush(**base, q=i["q"], k=i["k"], v=i["v"], a=i["a"], b=i["b"],
                 initial_state_source=S_ind_so, request_indices=sub,
                 flush_steps=P_sub, disable_state_update=False,
                 disable_output=True)
        torch.cuda.synchronize()

        # packed reference: gather the subset, state slots passed explicitly
        sl = sub.long()
        S_pk = i["state"].clone()
        o_pk = wy_flush(**base,
                        q=i["q"][sl].contiguous(), k=i["k"][sl].contiguous(),
                        v=i["v"][sl].contiguous(), a=i["a"][sl].contiguous(),
                        b=i["b"][sl].contiguous(),
                        initial_state_source=S_pk,
                        initial_state_indices=sub, flush_steps=P_sub,
                        disable_state_update=False)
        torch.cuda.synchronize()

        tag = f"K B={B_full} sub={n_sub}"
        if not torch.equal(S_ind, S_pk):
            fail(tag, "indirect fold != packed fold (bit)")
        if not torch.equal(S_ind_so, S_pk):
            fail(tag, "indirect state-only fold != packed fold (bit)")
        if not torch.equal(o_ind[sl], o_pk):
            fail(tag, "indirect outputs (at ring rows) != packed outputs (bit)")
        untouched = torch.ones(B_full, dtype=torch.bool, device=DEV)
        untouched[sl[P_sub > 0]] = False
        if not torch.equal(S_ind[untouched], i["state"][untouched]):
            fail(tag, "state slots outside the folding subset modified")

        # identity regression: explicit arange == default None (bit)
        S_a = i["state"].clone()
        P_all = torch.randint(0, 13, (B_full,), generator=g).to(torch.int32).to(DEV)
        o_a = wy_flush(**base, q=i["q"], k=i["k"], v=i["v"], a=i["a"], b=i["b"],
                       initial_state_source=S_a,
                       request_indices=torch.arange(
                           B_full, dtype=torch.int32, device=DEV),
                       initial_state_indices=i["idx"], flush_steps=P_all,
                       disable_state_update=False)
        S_n = i["state"].clone()
        o_n = wy_flush(**base, q=i["q"], k=i["k"], v=i["v"], a=i["a"], b=i["b"],
                       initial_state_source=S_n,
                       initial_state_indices=i["idx"], flush_steps=P_all,
                       disable_state_update=False)
        torch.cuda.synchronize()
        if not torch.equal(o_a, o_n) or not torch.equal(S_a, S_n):
            fail(tag, "explicit identity request_indices != default (bit)")
    print("K done", flush=True)


def main():
    torch.set_grad_enabled(False)
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
    for s in (section_a, section_b, section_c, section_d, section_e,
              section_f, section_g, section_h, section_i, section_j,
              section_k):
        s()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): {sorted(set(FAILURES))}")
        return 1
    print("\nALL SECTIONS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
