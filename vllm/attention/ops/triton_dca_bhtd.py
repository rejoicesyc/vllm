"""
The following kernels are adapted from flash-attention:
https://github.com/FlagOpen/FlagAttention/blob/ee91638dec6da8c00c4113d179f469e0ffcd5852/src/flag_attn/flash.py
"""

import pytest
import math

import torch
import triton
import triton.language as tl

from vllm_flash_attn.flash_attn_interface import (
    flash_attn_qkvpacked_func, 
    flash_attn_func,
    flash_attn_varlen_func,
)
from einops import rearrange, repeat


def maybe_contiguous(x):
    # only when the inner most dimension is contiguous can LDGSTS be used
    # so inner-dimension contiguity is enforced.
    return x.contiguous() if x.stride(-1) != 1 else x

def rounded_multiple(a, b):
    return (a + b - 1) // b * b

@triton.jit
def _fwd_inner(
    q_ptrs,
    k_ptrs,
    v_ptrs,
    sm_scale,
    qk_scale,
    lo_n,
    hi_n,
    STAGE: tl.constexpr,
    offs_m,
    offs_k,
    offs_n_base,
    stride_kn,
    stride_vn,
    P_SEQ,
    M,
    N,
    IS_CAUSAL: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    LARGER_M: tl.constexpr,
    input_dtype: tl.constexpr,
):
    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")

    #Dot I trick: to place q in registers, it saves shared memory
    # if BLOCK_DMODEL < 128:
    #     I = tl.where(offs_k[:, None] == offs_k,
    #                  tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
    #                  tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
    #     q = tl.dot(q, I).to(input_dtype)
    # else:
    #     I = tl.where(offs_m_base[:, None] == offs_m_base,
    #                  tl.full((BLOCK_M, BLOCK_M), 1.0, dtype=input_dtype),
    #                  tl.full((BLOCK_M, BLOCK_M), 0.0, dtype=input_dtype))
    #     q = tl.dot(I, q).to(input_dtype)

    hi_n = tl.multiple_of(hi_n, BLOCK_N)
    lo_n = tl.multiple_of(lo_n, BLOCK_N)
    
    for start_n in range(lo_n, hi_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        # -- load k, v --
        if DIVISIBLE_N:
            k = tl.load(k_ptrs + start_n * stride_kn, cache_modifier=".cg")
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs + start_n * stride_kn, mask=mask_n[None, :], cache_modifier=".cg")

        # -- compute qk ---
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k)

        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))
        if IS_CAUSAL and STAGE == 1:
            causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)

        # -- compute partial sumexpn before applying dropout
        p_sum = tl.sum(p, 1)

        # -- apply dropout --
        # if IS_DROPOUT:
        #     offs_rng = start_n + offs_rng_base
        #     pmask = tl.rand(seed, offs_rng, n_rounds=6) > dropout_p
        #     p *= pmask.to(tl.float32)

        # -- scale and update acc: acc *= alpha[:, None]--
        if DIVISIBLE_N:
            v = tl.load(v_ptrs + start_n * stride_vn, cache_modifier=".cg")
        else:
            mask_n = offs_n < N
            v = tl.load(v_ptrs + start_n * stride_vn, mask=mask_n[:, None], cache_modifier=".cg")

        acc *= alpha[:, None]
        acc += tl.dot(p.to(input_dtype), v)

        # -- update m_i and l_i --
        l_i = l_i * alpha + p_sum
        m_i = m_i_new
        # update pointers
        # k_ptrs += BLOCK_N * stride_kn
        # v_ptrs += BLOCK_N * stride_vn

    # write back l & o
    if IS_CAUSAL and LARGER_M:
        is_empty_line = (offs_m + P_SEQ) < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float("-inf"), m_i * sm_scale + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i * sm_scale + tl.log(l_i) # log(normalizer)

    return acc, l, m_i


configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in ([3, 4, 5])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


# @triton.autotune(list(filter(keep, configs)), key=["M", "D"])
@triton.jit
def _triton_dca_bhtd_kernel(
    Q, Q_succ, Q_inter, K, V, sm_scale,
    dropout_p,
    seed,
    offset,
    L, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, D, P_SEQ,
    chunk_len,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, IS_DROPOUT: tl.constexpr, LARGER_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    Q_succ += off_z * stride_qz + off_h * stride_qh
    Q_inter += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M # l's shape is (B, H, M)

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    if IS_DROPOUT:
        rowblock_base = off_z * H * M * N + off_h * M * N + start_m * BLOCK_M * N
        offs_rng_base = offset + rowblock_base
        offs_rng_base += tl.arange(0, BLOCK_M)[:, None] * N
        offs_rng_base += tl.arange(0, BLOCK_N)[None, :]

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    q_succ_ptrs = Q_succ + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    q_inter_ptrs = Q_inter + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok) # (BLOCK_M, BLOCK_DMODEL)
    l_ptrs = L + offs_m

    # NOTE: Loop-Bound-For-N
    # The indices in m-dimension that this block may access is in `[start_m * BLOCK_M, (start_m + 1) * BLOCK_M)`.
    # According to the rule of causal masking, then max index in n-dimension that this block may access
    # is `P_SEQ + (start_m + 1) * BLOCK_M`.
    # However, the upper bound of index in n-dimension should never exceed the sequence length of k/v(`P_SEQ + N_CTX`).
    # `P_SEQ + (start_m + 1) * BLOCK_M` may be larger than `N`.
    # At this case, there would be illegal memory access when loading k & v tiles
    # if mask_n is not applied for loading(only when `DIVISIBLE_N`` is true).
    # See also https://github.com/FlagOpen/FlagAttention/pull/8
    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N

    # loop over k, v and update accumulators
    offs_n_init = offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n_init[None, :] * stride_vn) # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk) # (BLOCK_N, BLOCK_DMODEL)

    # dual chunk attention
    acc_inter = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_inter = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_succ = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_succ = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_intra = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_intra = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    intra_end = hi
    intra_begin = ((start_m * BLOCK_M) // chunk_len) * chunk_len
    succ_end = intra_begin
    succ_begin = intra_begin - min(intra_begin, chunk_len)
    inter_end = succ_begin
    inter_begin = tl.zeros_like(inter_end)

    if intra_end > intra_begin:
        acc_intra, l_intra, m_intra = _fwd_inner(
            q_ptrs,
            k_ptrs,
            v_ptrs,
            sm_scale,
            qk_scale,
            intra_begin,
            intra_end,
            1,
            offs_m,
            offs_k,
            offs_n_base,
            stride_kn,
            stride_vn,
            P_SEQ,
            M,
            N,
            IS_CAUSAL,
            DIVISIBLE_M,
            DIVISIBLE_N,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            LARGER_M,
            input_dtype,
        )

    if succ_end > succ_begin:
        acc_succ, l_succ, m_succ = _fwd_inner(
            q_succ_ptrs,
            k_ptrs,
            v_ptrs,
            sm_scale,
            qk_scale,
            succ_begin,
            succ_end,
            2,
            offs_m,
            offs_k,
            offs_n_base,
            stride_kn,
            stride_vn,
            P_SEQ,
            M,
            N,
            IS_CAUSAL,
            DIVISIBLE_M,
            DIVISIBLE_N,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            LARGER_M,
            input_dtype,
        )

    if inter_end > inter_begin:
        acc_inter, l_inter, m_inter = _fwd_inner(
            q_inter_ptrs,
            k_ptrs,
            v_ptrs,
            sm_scale,
            qk_scale,
            inter_begin,
            inter_end,
            2,
            offs_m,
            offs_k,
            offs_n_base,
            stride_kn,
            stride_vn,
            P_SEQ,
            M,
            N,
            IS_CAUSAL,
            DIVISIBLE_M,
            DIVISIBLE_N,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            LARGER_M,
            input_dtype,
        )

    #merge attn outputs
    l_max = tl.maximum(tl.maximum(l_inter, l_succ), l_intra)
    l_inter = tl.exp(l_inter - l_max)
    l_succ = tl.exp(l_succ - l_max)
    l_intra = tl.exp(l_intra - l_max)
    l_sum = l_inter + l_succ + l_intra
    l_rcp = 1. / l_sum
    acc_inter *= (l_inter * l_rcp)[:, None]
    acc_succ *= (l_succ * l_rcp)[:, None]
    acc_intra *= (l_intra * l_rcp)[:, None]
    acc_o = acc_inter + acc_succ + acc_intra

    if DIVISIBLE_M:
        tl.store(o_ptrs, acc_o, cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        tl.store(o_ptrs, acc_o, mask=mask_m[:, None], cache_modifier=".cg")


def triton_dca_bhtd(q, q_succ, q_inter, k, v, causal, sm_scale, dropout_p, chunk_len, best_config=False):
    Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Dq == Dk == Dv, "feature size of q, k, v should be equal"
    assert Dk in {16, 32, 64, 128}

    B, H, M, D = q.shape
    N = k.shape[2]
    Hk, Hv = k.shape[1], v.shape[1]
    assert Hk == Hv, "num of heads in k and v should be equal"
    assert H % Hk == 0, "number of heads in q must be a multiple of that in k & v"
    num_groups = H // Hk

    P_SEQ = N - M
    assert P_SEQ == 0
    larger_m = M > N

    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D)

    # contiguity
    q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)

    # to work around https://github.com/openai/triton/issues/2441
    device = torch.cuda.device_of(q)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    with torch.cuda.device(device):
        # Dropout preparation.
        is_dropout = dropout_p > 0
        if is_dropout:
            offset_increment = B * H * M * N
            seed, offset = philox_cuda_seed_offset(offset_increment)
        else:
            seed, offset = 0, 0

        if D == 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 5, 4
        elif D == 128:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 4, 4
        else:
            raise ValueError('not tuned yet')

        divisible_m = M % BLOCK_M == 0
        divisible_n = N % BLOCK_N == 0
        # consider using 3d grid to avoid div & rem
        # grid = (triton.cdiv(M, BLOCK_M), H, B)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), H, B)
        o = torch.empty_like(q)
        L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)

        _triton_dca_bhtd_kernel[grid](
            q, q_succ, q_inter, k, v, sm_scale,
            dropout_p, seed, offset,
            L, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            B, H, M, N, D, P_SEQ, chunk_len, num_groups,
            BLOCK_DMODEL=D,
            IS_CAUSAL=True, IS_DROPOUT=is_dropout, LARGER_M=larger_m,
            DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, 
            num_warps=num_warps, num_stages=num_stages,
        )
        if best_config:
            print('_triton_dca_bhtd_kernel:', _triton_dca_bhtd_kernel.best_config)

    return o, L
