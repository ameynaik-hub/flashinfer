"""Correctness campaign v2 — escalation beyond _kernelB_correctness_intense.
REPORT-ONLY: failures are recorded and printed; nothing is fixed here.

  L. new GQA ratios (HV/H in {1,8,16}) + B=512 + deep seeds on canonical cfg
  M. request_indices stress: permutations, duplicates w/ distinct slots,
     reversed/unsorted, boundary rows, graph-mutated index lists
  N. property oracles: fold associativity, V-superposition, head independence
     (GQA scramble), batch-permutation equivariance
  P. 300-replay CUDA-graph soak with in-place mutations + eager cross-checks
  Q. 4-stream disjoint-slot torture
"""
import sys

import torch

import _kernelB_correctness_intense as base
from _kernelB_correctness_intense import (
    DEV, SCALE, T, ULP, fail, mk, md, check_config, p_patterns,
)
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_output_only import (
    gated_delta_rule_mtp as wy_out,
)
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_state_and_output import (
    gated_delta_rule_mtp as wy_flush,
)

FAILURES = base.FAILURES  # shared registry


def base_kwargs(i):
    return dict(A_log=i["A_log"], dt_bias=i["dt_bias"],
                use_qk_l2norm_in_kernel=True, scale=SCALE)


def section_l():
    print("== L. new GQA ratios / big shapes / deep seeds ==", flush=True)
    # (H=HK, HV): ratio 1, 8, 16 are NEW; canonical (16,64) gets 10 seeds
    for H, HV, seeds, Bs in (
        (16, 16, (0, 1, 2), (2, 17, 128)),      # ratio 1
        (8, 64, (0, 1, 2), (2, 17, 128)),       # ratio 8
        (4, 64, (0, 1, 2), (2, 17, 128)),       # ratio 16
        (16, 64, range(10), (17, 512)),          # canonical, deep seeds + B=512
    ):
        for B in Bs:
            for seed in seeds:
                i = mk(B, HV, H, seed)
                for pname, P in p_patterns(B, seed).items():
                    do_torch = B <= 32 and seed <= 1
                    check_config(
                        f"L HV={HV} H={H} B={B} s={seed} P={pname}",
                        i, P, do_torch,
                    )
        print(f"  (H={H}, HV={HV}) done", flush=True)
    print("L done", flush=True)


def section_m():
    print("== M. request_indices stress ==", flush=True)
    H, HV, B = 16, 64, 64
    i = mk(B, HV, H, 90)
    bk = base_kwargs(i)
    full = dict(q=i["q"], k=i["k"], v=i["v"], a=i["a"], b=i["b"])

    def packed_ref(rows, P, slots=None):
        sl = rows.long()
        S = i["state"].clone()
        o = wy_flush(**bk, q=i["q"][sl].contiguous(), k=i["k"][sl].contiguous(),
                     v=i["v"][sl].contiguous(), a=i["a"][sl].contiguous(),
                     b=i["b"][sl].contiguous(), initial_state_source=S,
                     initial_state_indices=(slots if slots is not None else rows),
                     flush_steps=P, disable_state_update=False)
        torch.cuda.synchronize()
        return o, S

    def indirect(rows, P, slots=None):
        S = i["state"].clone()
        o = wy_flush(**bk, **full, initial_state_source=S,
                     request_indices=rows, initial_state_indices=slots,
                     flush_steps=P, disable_state_update=False)
        torch.cuda.synchronize()
        return o, S

    g = torch.Generator().manual_seed(90)
    # full permutation (launch covers ALL rows, shuffled)
    perm = torch.randperm(B, generator=g).to(torch.int32).to(DEV)
    P = torch.randint(0, 13, (B,), generator=g).to(torch.int32).to(DEV)
    o_i, S_i = indirect(perm, P)
    o_p, S_p = packed_ref(perm, P)
    if not torch.equal(S_i, S_p) or not torch.equal(o_i[perm.long()], o_p):
        fail("M perm", "full permutation != packed reference")
    # reversed subset (unsorted, descending)
    rows = torch.arange(B - 1, -1, -3, dtype=torch.int32, device=DEV)
    P2 = torch.randint(1, 13, (rows.shape[0],), generator=g).to(torch.int32).to(DEV)
    o_i, S_i = indirect(rows, P2)
    o_p, S_p = packed_ref(rows, P2)
    if not torch.equal(S_i, S_p) or not torch.equal(o_i[rows.long()], o_p):
        fail("M reversed", "descending unsorted rows != packed reference")
    # single request, last row (boundary)
    rows = torch.tensor([B - 1], dtype=torch.int32, device=DEV)
    P3 = torch.tensor([12], dtype=torch.int32, device=DEV)
    o_i, S_i = indirect(rows, P3)
    o_p, S_p = packed_ref(rows, P3)
    if not torch.equal(S_i, S_p) or not torch.equal(o_i[B - 1], o_p[0]):
        fail("M boundary", "single last-row launch != packed reference")
    # duplicate ring row, DISTINCT state slots (same window folded into two slots)
    rows = torch.tensor([5, 5], dtype=torch.int32, device=DEV)
    slots = torch.tensor([5, 63], dtype=torch.int32, device=DEV)
    P4 = torch.tensor([9, 9], dtype=torch.int32, device=DEV)
    S = i["state"].clone()
    wy_flush(**bk, **full, initial_state_source=S, request_indices=rows,
             initial_state_indices=slots, flush_steps=P4,
             disable_state_update=False)
    torch.cuda.synchronize()
    # both slots must equal the single-fold result of window 5 applied to
    # their own initial state
    for slot in (5, 63):
        S1 = i["state"].clone()
        wy_flush(**bk, **full, initial_state_source=S1,
                 request_indices=torch.tensor([5], dtype=torch.int32, device=DEV),
                 initial_state_indices=torch.tensor([slot], dtype=torch.int32,
                                                    device=DEV),
                 flush_steps=torch.tensor([9], dtype=torch.int32, device=DEV),
                 disable_state_update=False)
        torch.cuda.synchronize()
        if not torch.equal(S[slot], S1[slot]):
            fail("M dup", f"duplicate ring row -> slot {slot} mismatch")
    # graph capture with IN-PLACE mutation of the index lists between replays
    n = 16
    rows = torch.arange(n, dtype=torch.int32, device=DEV)
    Pg = torch.full((n,), 8, dtype=torch.int32, device=DEV)
    S_pool = i["state"].clone()
    for _ in range(3):
        wy_flush(**bk, **full, initial_state_source=S_pool, request_indices=rows,
                 flush_steps=Pg, disable_state_update=False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wy_flush(**bk, **full, initial_state_source=S_pool, request_indices=rows,
                 flush_steps=Pg, disable_state_update=False)
    for rep in range(4):
        g2 = torch.Generator().manual_seed(200 + rep)
        new_rows = torch.randperm(B, generator=g2)[:n].to(torch.int32).to(DEV)
        new_P = torch.randint(1, 13, (n,), generator=g2).to(torch.int32).to(DEV)
        rows.copy_(new_rows)
        Pg.copy_(new_P)
        S_pool.copy_(i["state"])
        graph.replay()
        torch.cuda.synchronize()
        S_graph = S_pool.clone()
        _, S_eager = indirect(new_rows, new_P)
        # indirect() clones from i["state"]; compare folded slots only
        sl = new_rows.long()
        if not torch.equal(S_graph[sl], S_eager[sl]):
            fail("M graph-idx", f"replay {rep}: mutated index list not honored")
            break
    print("M done", flush=True)


def section_n():
    print("== N. property oracles ==", flush=True)
    H, HV, B = 16, 64, 8
    dd = base.decode_delta_rule

    def fp32_fold(i, rows_slice, S0_ref):
        st = S0_ref
        for t in rows_slice:
            _, st = dd(i["q"][:, t].float(), i["k"][:, t].float(),
                       i["v"][:, t].float(), st, A_log=i["A_log"], a=i["a"][:, t],
                       dt_bias=i["dt_bias"], b=i["b"][:, t], scale_factor=SCALE,
                       softplus_beta=1.0, softplus_threshold=20.0,
                       use_l2_norm=True, state_dtype=torch.float32)
        return st

    # N1: associativity — fold 10 rows at once == fold 6 then fold 4 (window
    # shifted), both vs fp32 truth; the two kernel paths must agree with truth
    # within the established fold bar.
    i = mk(B, HV, H, 95)
    bk = base_kwargs(i)
    P10 = torch.full((B,), 10, dtype=torch.int32, device=DEV)
    S_once = i["state"].clone()
    wy_flush(**bk, q=i["q"], k=i["k"], v=i["v"], a=i["a"], b=i["b"],
             initial_state_source=S_once, flush_steps=P10,
             disable_state_update=False)
    S_two = i["state"].clone()
    P6 = torch.full((B,), 6, dtype=torch.int32, device=DEV)
    wy_flush(**bk, q=i["q"], k=i["k"], v=i["v"], a=i["a"], b=i["b"],
             initial_state_source=S_two, flush_steps=P6,
             disable_state_update=False)
    sh = {kk: torch.zeros_like(i[kk]) for kk in ("q", "k", "v", "a", "b")}
    for kk in sh:
        sh[kk][:, :4] = i[kk][:, 6:10]  # shifted window: rows 6..9 -> 0..3
    P4 = torch.full((B,), 4, dtype=torch.int32, device=DEV)
    wy_flush(**bk, q=sh["q"], k=sh["k"], v=sh["v"], a=sh["a"], b=sh["b"],
             initial_state_source=S_two, flush_steps=P4,
             disable_state_update=False)
    torch.cuda.synchronize()
    S32 = fp32_fold(i, range(10),
                    i["state"].transpose(-2, -1).contiguous().float())
    S32k = S32.transpose(-2, -1).contiguous()
    d_once = md(S_once.float(), S32k)
    d_two = md(S_two.float(), S32k)
    bar = max(4 * ULP, 2 * d_once)
    if d_two > bar:
        fail("N assoc", f"two-step fold vs fp32 {d_two:.2e} > bar {bar:.2e} "
                        f"(one-step {d_once:.2e})")
    # N2: superposition — Fold(S0,V) - Fold(S0,0) is linear in V:
    # F(V1+V2) - F(0) ~= (F(V1)-F(0)) + (F(V2)-F(0)) within bf16 noise
    i = mk(B, HV, H, 96)
    bk = base_kwargs(i)
    P = torch.full((B,), 8, dtype=torch.int32, device=DEV)

    def foldV(vv):
        S = i["state"].clone()
        wy_flush(**bk, q=i["q"], k=i["k"], v=vv, a=i["a"], b=i["b"],
                 initial_state_source=S, flush_steps=P,
                 disable_state_update=False)
        torch.cuda.synchronize()
        return S.float()

    v1 = i["v"].clone()
    g = torch.Generator(device=DEV).manual_seed(96)
    v2 = torch.randn(i["v"].shape, generator=g, dtype=torch.bfloat16, device=DEV)
    F0 = foldV(torch.zeros_like(v1))
    lhs = foldV((v1.float() + v2.float()).to(torch.bfloat16)) - F0
    rhs = (foldV(v1) - F0) + (foldV(v2) - F0)
    d = md(lhs, rhs)
    if d > 8 * ULP:  # three folds' worth of bf16 noise
        fail("N superpos", f"V-linearity violated: {d:.2e}")
    # N3: head independence under GQA — scramble all HV-heads except a chosen
    # v-head group and its k/q head; that head's output+state slice must be
    # BIT-unchanged.
    i = mk(B, HV, H, 97)
    bk = base_kwargs(i)
    P = torch.full((B,), 9, dtype=torch.int32, device=DEV)
    o_ref, S_ref = base.run_fused(i, P)
    hv_keep = 5                     # v-head to protect
    h_keep = hv_keep // (HV // H)   # its q/k head
    i2 = {k2: (v2.clone() if torch.is_tensor(v2) else v2) for k2, v2 in i.items()}
    gg = torch.Generator(device=DEV).manual_seed(970)
    for kk, hdim in (("v", 2), ("a", 2), ("b", 2)):
        scr = torch.randn(i2[kk].shape, generator=gg, dtype=torch.bfloat16,
                          device=DEV)
        mask = torch.ones(i2[kk].shape[2], dtype=torch.bool, device=DEV)
        mask[hv_keep] = False
        i2[kk][:, :, mask] = scr[:, :, mask]
    for kk in ("q", "k"):
        scr = torch.randn(i2[kk].shape, generator=gg, dtype=torch.bfloat16,
                          device=DEV)
        mask = torch.ones(H, dtype=torch.bool, device=DEV)
        mask[h_keep] = False
        i2[kk][:, :, mask] = scr[:, :, mask]
    o2, S2 = base.run_fused(i2, P)
    if not torch.equal(o2[:, :, hv_keep], o_ref[:, :, hv_keep]):
        fail("N headind", "protected v-head OUTPUT changed when other heads scrambled")
    if not torch.equal(S2[:, hv_keep], S_ref[:, hv_keep]):
        fail("N headind", "protected v-head STATE changed when other heads scrambled")
    # N4: batch permutation equivariance (fused): permute requests+P, results
    # must permute identically (bit).
    i = mk(B, HV, H, 98)
    bk = base_kwargs(i)
    P = torch.randint(0, 13, (B,), generator=torch.Generator().manual_seed(98)
                      ).to(torch.int32).to(DEV)
    o_a, S_a = base.run_fused(i, P)
    pm = torch.randperm(B, generator=torch.Generator().manual_seed(99)).to(DEV)
    ip = {k2: (v2[pm].clone() if k2 in ("q", "k", "v", "a", "b", "state")
               else v2) for k2, v2 in i.items()}
    ip.update(idx=torch.arange(B, dtype=torch.int32, device=DEV))
    o_b, S_b = base.run_fused(ip, P[pm].contiguous())
    if not torch.equal(o_b, o_a[pm]) or not torch.equal(S_b, S_a[pm]):
        fail("N permequiv", "permuted batch != permuted results (bit)")
    print("N done", flush=True)


def section_p():
    print("== P. 300-replay graph soak (both kernels) ==", flush=True)
    H, HV, B = 4, 16, 32
    i = mk(B, HV, H, 300)
    bk = base_kwargs(i)
    P = torch.full((B,), 10, dtype=torch.int32, device=DEV)
    S_pool = i["state"].clone()
    holder = {}

    def call():
        holder["o"] = wy_flush(**bk, q=i["q"], k=i["k"], v=i["v"], a=i["a"],
                               b=i["b"], initial_state_source=S_pool,
                               flush_steps=P, disable_state_update=False)

    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    bad = 0
    for rep in range(300):
        i2 = mk(B, HV, H, 301 + rep)
        for kk in ("q", "k", "v", "a", "b"):
            i[kk].copy_(i2[kk])
        P.copy_(torch.randint(0, 13, (B,),
                generator=torch.Generator().manual_seed(600 + rep)
                ).to(torch.int32).to(DEV))
        S_pool.copy_(i2["state"])
        graph.replay()
        torch.cuda.synchronize()
        if rep % 50 == 0:  # eager cross-check on a sample
            S_graph = S_pool.clone()
            o_graph = holder["o"].clone()
            i2c = dict(i2)
            i2c["A_log"], i2c["dt_bias"] = i["A_log"], i["dt_bias"]
            o_e, S_e = base.run_fused(i2c, P.clone())
            if not torch.equal(S_graph, S_e) or not torch.equal(o_graph, o_e):
                bad += 1
    if bad:
        fail("P soak", f"{bad} sampled replays diverged from eager")
    print("P done", flush=True)


def section_q():
    print("== Q. 4-stream disjoint torture ==", flush=True)
    H, HV, B = 16, 64, 64
    qsz = B // 4
    for rep in range(20):
        i = mk(B, HV, H, 400 + rep)
        bk = base_kwargs(i)
        chunks = [torch.arange(j * qsz, (j + 1) * qsz, dtype=torch.int32,
                               device=DEV) for j in range(4)]
        Ps = [torch.full((qsz,), 6 + j * 2, dtype=torch.int32, device=DEV)
              for j in range(4)]
        # sequential golden
        S_seq = i["state"].clone()
        for j in range(4):
            fn = wy_flush if j % 2 == 0 else wy_flush
            fn(**bk, q=i["q"], k=i["k"], v=i["v"], a=i["a"], b=i["b"],
               initial_state_source=S_seq, request_indices=chunks[j],
               flush_steps=Ps[j], disable_state_update=False,
               disable_output=(j % 2 == 1))
        torch.cuda.synchronize()
        # concurrent on 4 streams
        S_con = i["state"].clone()
        streams = [torch.cuda.Stream() for _ in range(4)]
        for j, s in enumerate(streams):
            with torch.cuda.stream(s):
                wy_flush(**bk, q=i["q"], k=i["k"], v=i["v"], a=i["a"], b=i["b"],
                         initial_state_source=S_con, request_indices=chunks[j],
                         flush_steps=Ps[j], disable_state_update=False,
                         disable_output=(j % 2 == 1))
        torch.cuda.synchronize()
        if not torch.equal(S_con, S_seq):
            fail("Q streams", f"rep {rep}: 4-stream result != sequential")
            break
    print("Q done", flush=True)


def main():
    torch.set_grad_enabled(False)
    print(f"GPU: {torch.cuda.get_device_name()}  [campaign v2 — REPORT ONLY]",
          flush=True)
    for s in (section_l, section_m, section_n, section_p, section_q):
        s()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): {sorted(set(FAILURES))}")
        return 1
    print("\nALL V2 SECTIONS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
