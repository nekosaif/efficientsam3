"""Isolate the bug to nn.MultiheadAttention.

Build two MHA modules with byte-identical weights/biases, run forward on the
same input, compare. If outputs differ -> the bug is in MHA module-state
that survives state_dict round-trip (e.g., a lazy-init buffer, a hook, a
fast-path flag set on first forward, etc.).
"""

from __future__ import annotations

import torch
import torch.nn as nn

torch.manual_seed(42)


def diff(a, b, name):
    d = float((a.float() - b.float()).abs().max().item())
    eq = '✓' if torch.equal(a, b) else '✗'
    print(f'  {name:40} max_diff={d:.6e}  equal={eq}  dtype={a.dtype}')


def main():
    device = 'cuda'
    embed_dim, num_heads, dropout = 256, 8, 0.0
    B, Lq, Lkv = 1, 64, 5184

    # Two MHA instances. After building, A is "trained" with a synthetic step.
    mha_A = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True).to(device)
    mha_B = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True).to(device)

    # Optimizer on A only, train one step so its params change
    opt = torch.optim.AdamW(mha_A.parameters(), lr=0.5)
    q_tr = torch.randn(B, Lq, embed_dim, device=device, requires_grad=False)
    kv_tr = torch.randn(B, Lkv, embed_dim, device=device, requires_grad=False)
    target = torch.randn(B, Lq, embed_dim, device=device)
    mha_A.train()
    out, _ = mha_A(q_tr, kv_tr, kv_tr, need_weights=False)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()

    # Now copy A's weights into B (so they have IDENTICAL params)
    mha_B.load_state_dict(mha_A.state_dict(), strict=True)

    # Verify byte-identical params
    print('=== Param byte-identity ===')
    for name, p_a in mha_A.named_parameters():
        p_b = dict(mha_B.named_parameters())[name]
        diff(p_a, p_b, name)

    # Run forward in eval mode under bf16 autocast on same input
    print('\n=== Forward comparison ===')
    mha_A.eval(); mha_B.eval()
    torch.manual_seed(123)
    q = torch.randn(B, Lq, embed_dim, device=device)
    kv = torch.randn(B, Lkv, embed_dim, device=device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out_A, _ = mha_A(q, kv, kv, need_weights=False)
            out_B, _ = mha_B(q, kv, kv, need_weights=False)
    diff(out_A, out_B, 'cross_attn(q,kv,kv) [w/ no_grad + bf16 autocast]')

    # Run forward in float32, no autocast
    print('\n=== Forward comparison (no autocast, fp32) ===')
    with torch.no_grad():
        out_A2, _ = mha_A(q, kv, kv, need_weights=False)
        out_B2, _ = mha_B(q, kv, kv, need_weights=False)
    diff(out_A2, out_B2, 'cross_attn(q,kv,kv) [fp32]')

    # Repeated forwards on the SAME module — determinism check
    print('\n=== Repeated forwards same module (determinism) ===')
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(3):
                o, _ = mha_A(q, kv, kv, need_weights=False)
                d = float((o.float() - out_A.float()).abs().max().item())
                print(f'  fwd_A[{i+1}] diff vs out_A: {d:.6e}')

    # Compare instance attributes
    print('\n=== Instance attribute diff ===')
    a_dict = mha_A.__dict__
    b_dict = mha_B.__dict__
    a_keys = set(a_dict.keys()); b_keys = set(b_dict.keys())
    for k in sorted(a_keys ^ b_keys):
        print(f'  key only in {"A" if k in a_keys else "B"}: {k}')
    for k in sorted(a_keys & b_keys):
        va, vb = a_dict[k], b_dict[k]
        if torch.is_tensor(va) or torch.is_tensor(vb):
            continue  # handled above
        if va != vb:
            print(f'  {k}: A={va!r}  vs  B={vb!r}')
    print('  (only differences printed)')

    # Force the slow path on both — does that converge?
    print('\n=== Force training=True to disable fast path? ===')
    mha_A.train(); mha_B.train()
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out_At, _ = mha_A(q, kv, kv, need_weights=False)
            out_Bt, _ = mha_B(q, kv, kv, need_weights=False)
    diff(out_At, out_Bt, 'cross_attn in train()')

    # Use torch.backends.cuda.sdp_kernel context to force math backend
    print('\n=== Force MATH backend ===')
    mha_A.eval(); mha_B.eval()
    try:
        with torch.no_grad():
            with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out_Am, _ = mha_A(q, kv, kv, need_weights=False)
                    out_Bm, _ = mha_B(q, kv, kv, need_weights=False)
        diff(out_Am, out_Bm, 'cross_attn force math backend')
    except Exception as e:
        print('  sdp_kernel context unsupported here:', e)


if __name__ == '__main__':
    main()
