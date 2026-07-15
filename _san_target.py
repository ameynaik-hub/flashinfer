"""Tiny launches of every kernel path for compute-sanitizer (B=2, HV=16)."""
import math
import torch
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_output_only import (
    gated_delta_rule_mtp as wy_out,
)
from flashinfer.gdn_kernels.gdn_decode_bf16_wy_state_and_output import (
    gated_delta_rule_mtp as wy_flush,
)
torch.set_grad_enabled(False)
H, HV, K, V, B = 4, 16, 128, 128, 2
d = "cuda"
def tok(t):
    return dict(q=torch.randn(B, t, H, K, dtype=torch.bfloat16, device=d),
                k=torch.randn(B, t, H, K, dtype=torch.bfloat16, device=d),
                v=torch.randn(B, t, HV, V, dtype=torch.bfloat16, device=d),
                a=torch.randn(B, t, HV, dtype=torch.bfloat16, device=d) * 0.1,
                b=torch.randn(B, t, HV, dtype=torch.bfloat16, device=d))
A_log = torch.randn(HV, device=d) * 0.1
dtb = torch.randn(HV, device=d) * 0.1
st = torch.randn(B, HV, V, K, dtype=torch.bfloat16, device=d)
idx = torch.arange(B, dtype=torch.int32, device=d)
P = torch.tensor([12, 0], dtype=torch.int32, device=d)
sub = torch.tensor([1], dtype=torch.int32, device=d)
cm = dict(A_log=A_log, dt_bias=dtb, use_qk_l2norm_in_kernel=True,
          scale=1 / math.sqrt(K))
wy_out(**tok(4), **cm, initial_state_source=st, initial_state_indices=idx,
       disable_state_update=True)                       # native T=4
wy_out(**tok(16), **cm, initial_state_source=st, initial_state_indices=idx,
       disable_state_update=True)                       # 16-row window
wy_out(**tok(16), **cm, initial_state_source=st, request_indices=sub,
       disable_state_update=True)                       # indirect subset
wy_flush(**tok(16), **cm, initial_state_source=st, flush_steps=P,
         disable_state_update=False)                    # kernel B fused
wy_flush(**tok(16), **cm, initial_state_source=st, flush_steps=P,
         disable_state_update=False, disable_output=True)  # state-only
torch.cuda.synchronize()
print("SAN_TARGET_DONE")
