# PR #2727 Review Notes — Non-Contiguous State for GDN Decode Pretranspose

**Reviewer:** ameyn (AI-assisted with Claude Code)
**Date:** 2026-03-11
**GPU:** B200 (SM100, Blackwell)

## What the PR Does

Eliminates the `.contiguous()` copy for non-contiguous `initial_state` tensors in GDN decode pretranspose. vLLM's page-strided state pool creates non-contiguous 4D tensors; previously the code forced a reshape to 3D `[pool_size*HV, V, K]` which required contiguous memory.

### Key changes (3 files):

1. **`flashinfer/gdn_decode.py`**: Pool path now passes 4D `[pool_size, HV, V, K]` tensor directly instead of reshaping to 3D. Adds `stride(-1) == 1` safety check. Removes the `.is_contiguous()` assertion.
2. **`flashinfer/gdn_kernels/gdn_decode_pretranspose.py`**: Both `small_batch` and `big_batch` kernels branch on `const_expr(use_pool_indexing)` to use 4D indexing (pool path) vs 3D (direct path). Writeback tiling updated from 4D to 5D for pool path. `from_dlpack` replaces `make_fake_compact_tensor` to preserve stride info. Cache key now includes pool strides.
3. **`tests/gdn/test_decode_pretranspose_noncontiguous_pool.py`**: New test with page_gap=[2,3].

## Review Checks

### Check 1 — Non-pool pretranspose regression: 10/10 PASSED

```bash
source /home/scratch.ameyn_gpu_2/cudeepy/cudeepy_venv/bin/activate && \
FLASHINFER_WORKSPACE_BASE=/tmp/claude-102806/flashinfer_cache \
pytest tests/gdn/test_decode_delta_rule.py \
  -k "pretranspose and not pool and not mtp and not bf16_state and not noncontiguous" \
  -x -v --tb=short -rs 2>&1 | tail -40
```

### Check 2 — Contiguous pool + negative indices + all-padding regression: 17/17 PASSED

```bash
source /home/scratch.ameyn_gpu_2/cudeepy/cudeepy_venv/bin/activate && \
FLASHINFER_WORKSPACE_BASE=/tmp/claude-102806/flashinfer_cache \
pytest tests/gdn/test_decode_delta_rule.py \
  -k "pool and not noncontiguous" \
  -x -v --tb=short -rs 2>&1 | tail -40
```

### Check 3 — Non-contiguous pool (PR's tests + added tests): 6/6 PASSED

```bash
source /home/scratch.ameyn_gpu_2/cudeepy/cudeepy_venv/bin/activate && \
FLASHINFER_WORKSPACE_BASE=/tmp/claude-102806/flashinfer_cache \
pytest tests/gdn/test_decode_pretranspose_noncontiguous_pool.py \
  tests/gdn/test_decode_delta_rule.py \
  -k "noncontiguous" -x -v --tb=short -rs 2>&1 | tail -30
```

### Check 4 — Code review: PASS

**Writeback dimensionality — CORRECT:**
- Pool path (small batch kernel, line 145-146): `gDst = cute.local_tile(h0_source, (1, 1, TILE_V, TILE_K), (pool_idx, i_hv, None, 0))` — tiling a 4D `[pool_size, HV, V, K]` tensor produces 5D. Writeback uses `(1, 1, 1, vec_size, 1)` with offsets `(0, 0, row+row_offset, lane_id, v_tiles)` — 5D, correct.
- Non-pool path: stays 3D→4D with `(1, TILE_V, TILE_K)` tiling, writeback `(1, 1, vec_size, 1)` — 4D, correct.
- Both `big_batch` and `small_batch` kernels have symmetric changes — confirmed.

**`from_dlpack` change:** Replaced `make_fake_compact_tensor` (which assumed contiguous layout) with `from_dlpack` which preserves actual stride information from non-contiguous PyTorch tensors. The CuTe compiler sees the real strides and generates correct indexing. Correct approach.

**Cache key:** Now includes `pool_size` + all 4 strides, ensuring different pool layouts trigger separate kernel compilations.

**Safety check:** `stride(-1) == 1` assertion at `gdn_decode.py:195` guards that the kernel assumes K-contiguous layout.

**Note:** Every unique stride pattern triggers a separate kernel compilation. For vLLM's use case this is fine (pool stride is stable across requests).

### Summary: 33/33 tests passed on B200 (SM100)

| Check | Description | Result |
|-------|-------------|--------|
| 1 | Non-pool pretranspose (batch 1-512) | 10/10 PASSED |
| 2 | Contiguous pool + negative indices + all-padding | 17/17 PASSED |
| 3 | Non-contiguous pool (PR's 2 + 4 added tests) | 6/6 PASSED |
| 4 | Kernel writeback dimensionality code review | PASS |

## Benchmark Results (B200, Qwen3-Next config)

```bash
source /home/scratch.ameyn_gpu_2/cudeepy/cudeepy_venv/bin/activate && \
FLASHINFER_WORKSPACE_BASE=/tmp/claude-102806/flashinfer_cache \
python benchmarks/bench_gdn_decode.py \
  --version all --batch-size 1 8 32 128 512 --preset qwen3-next 2>&1
```

| Batch | FI-PreTranspose (us) | FI-NonTranspose (us) | Triton-PreTr (us) | KlastBf16 (us) | FI/TR Speedup |
|------:|---------------------:|---------------------:|-------------------:|----------------:|--------------:|
| 1     | 3.74                 | 5.06                 | 5.86               | 7.62            | 1.56x         |
| 8     | 7.65                 | 12.06                | 9.84               | 11.07           | 1.29x         |
| 32    | 22.91                | 32.70                | 31.54              | 20.27           | 1.38x         |
| 128   | 81.57                | 107.78               | 113.94             | 52.83           | 1.40x         |
| 512   | 314.17               | 410.82               | 440.32             | 182.56          | 1.40x         |

**No performance regressions.** FlashInfer Pretranspose averages **1.41x** faster than Triton Pretranspose — consistent with pre-PR numbers. The non-contiguous stride support adds no measurable overhead to the contiguous path.

## Minor Nit

- **Dead code on lines 706 and 809** of `flashinfer/gdn_kernels/gdn_decode_pretranspose.py`: The expression `v_dim * k_dim * 4 / 1024 / 1024` is a dead expression (no assignment, no side effect). Pre-existing (changed from `v_dim * k_dim * batch_size * 4 / 1024 / 1024` by the PR, but was already dead before).

## Files Read During Review

- `flashinfer/gdn_kernels/gdn_decode_pretranspose.py` — kernel implementation (main review target)
- `flashinfer/gdn_decode.py` — Python API (pool path changes)
- `tests/gdn/test_decode_delta_rule.py` — existing tests
- `benchmarks/bench_gdn_decode.py` — benchmark script

## Files Modified

- `tests/gdn/test_decode_delta_rule.py` — added `test_decode_kernel_pretranspose_pool_noncontiguous` (4 parametrized cases: batch_size=[4,16] x page_gap=[2,3])
