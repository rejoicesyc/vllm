import random
import pytest
import time

import torch
from vllm.attention.ops.triton_dca_bhtd import triton_dca_bhtd

from einops import rearrange

from vllm_flash_attn.flash_attn_interface import (
    flash_attn_qkvpacked_func, 
    flash_attn_func,
    flash_attn_varlen_func,
)

"""
reference implement from https://github.com/vllm-project/vllm/pull/6139
"""

def _bruteforce_dynamic_chunk_flash_attn_func(
    q,
    q_succ,
    q_inter,
    k,
    v,
    block_table,
    softmax_scale,
    chunk_size,
    local_size,
    original_max_position_embeddings,
    current_prefill_original_seq_lens_tensor,
    k_length,
):

    def do_flash_attn(
        query_states,
        key_states,
        value_states,
        causal=True,
        block_table=None,
        max_seqlen_k=None,
    ):
        if max_seqlen_k is None:
            max_seqlen_k = key_states.shape[0]

        output, softmax_lse, _ = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            softmax_scale=softmax_scale,
            cu_seqlens_q=torch.tensor(
                [0, query_states.shape[0]],
                dtype=torch.int32,
                device=query_states.device,
            ),
            max_seqlen_q=query_states.shape[0],
            cu_seqlens_k=torch.tensor(
                [0, max_seqlen_k],
                dtype=torch.int32,
                device=query_states.device,
            ),
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            block_table=block_table,
            return_attn_probs=True,
        )
        return output, softmax_lse

    def merge_attn_outputs(flash_results):
        attn_outputs_all = []
        print(flash_results)
        for flash_per_chunk in flash_results:
            if len(flash_per_chunk) == 1:
                attn_outputs_all.append(flash_per_chunk[0][0])
                continue
            attn_outputs = torch.stack([
                flash_attn_output[0]
                for flash_attn_output in flash_per_chunk
            ])
            logits = torch.stack([
                flash_attn_output[1]
                for flash_attn_output in flash_per_chunk
            ]).to(torch.float32)
            max_logits = torch.max(logits, dim=0).values
            stable_logits = logits - max_logits.unsqueeze(0)
            lse_s = torch.exp(stable_logits).detach()
            lse_sum = torch.sum(lse_s, dim=0)
            lse_s /= lse_sum
            attn_outputs *= lse_s.unsqueeze(-1).transpose(2, 3).squeeze(1)
            attn_outputs_all.append(attn_outputs.sum(dim=0))
        return torch.cat(attn_outputs_all, dim=0)

    def get_block(begin, end):
        return block_table[:,
                            begin // block_size:(end - 1) // block_size + 1]

    flash_results = []
    chunk_len = chunk_size - local_size
    if block_table is not None:
        block_size = v.shape[1]
        if chunk_len % block_size != 0:
            raise ValueError("chunk_len must be divisible by block_size.")
    else:
        block_size = 1

    if original_max_position_embeddings > 0:
        mscale = max(
            0.1 * (current_prefill_original_seq_lens_tensor[0] /
                    original_max_position_embeddings).log() + 1.0,
            1.0,
        )
        softmax_scale = softmax_scale * mscale

    begin = k_length - q.shape[0]

    while begin < k_length:
        flash_per_chunk = []

        prev_chunk_end_pos = (begin // chunk_len) * chunk_len
        next_chunk_end_pos = prev_chunk_end_pos + chunk_len
        end = min(next_chunk_end_pos, k_length)
        qbegin = begin - (k_length - q.shape[0])
        qend = end - (k_length - q.shape[0])

        q_states_intra = q[qbegin:qend]
        if block_table is not None:
            block_table_intra = get_block(prev_chunk_end_pos, end)
            flash_result = do_flash_attn(
                q_states_intra,
                k,
                v,
                block_table=block_table_intra,
                max_seqlen_k=end - prev_chunk_end_pos,
            )
        else:
            print(f'intra flash q[{qbegin}:{qend}] kv[{prev_chunk_end_pos}:{end}]')
            k_states_intra = k[prev_chunk_end_pos:end]
            v_states_intra = v[prev_chunk_end_pos:end]
            flash_result = do_flash_attn(q_states_intra, k_states_intra,
                                            v_states_intra)
        flash_per_chunk.append(flash_result)

        if prev_chunk_end_pos - chunk_len >= 0:
            q_states_succ = q_succ[qbegin:qend]
            if block_table is not None:
                block_table_succ = get_block(
                    prev_chunk_end_pos - chunk_len, prev_chunk_end_pos)
                flash_result = do_flash_attn(
                    q_states_succ,
                    k,
                    v,
                    False,
                    block_table=block_table_succ,
                    max_seqlen_k=chunk_len,
                )
            else:
                k_states_succ = k[prev_chunk_end_pos -
                                    chunk_len:prev_chunk_end_pos]
                v_states_succ = v[prev_chunk_end_pos -
                                    chunk_len:prev_chunk_end_pos]
                flash_result = do_flash_attn(q_states_succ, k_states_succ,
                                                v_states_succ, False)
            flash_per_chunk.append(flash_result)

        if prev_chunk_end_pos - chunk_len * 2 >= 0:
            q_states_inter = q_inter[qbegin:qend]

            if block_table is not None:
                block_table_inter = get_block(
                    0, prev_chunk_end_pos - chunk_len)
                flash_result = do_flash_attn(
                    q_states_inter,
                    k,
                    v,
                    False,
                    block_table=block_table_inter,
                    max_seqlen_k=prev_chunk_end_pos - chunk_len,
                )
            else:
                k_states_inter = k[:prev_chunk_end_pos - chunk_len]
                v_states_inter = v[:prev_chunk_end_pos - chunk_len]
                flash_result = do_flash_attn(q_states_inter,
                                                k_states_inter,
                                                v_states_inter, False)
            flash_per_chunk.append(flash_result)

        begin = end
        flash_results.append(flash_per_chunk)

    attn_output = merge_attn_outputs(flash_results)

    return attn_output


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [
    (1, 48, 2048, 64),
    (1, 48, 2048, 128),
    (1, 48, 32768, 64),
])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize('chunk_size,local_size', [(512, 0), (8192, 1024)])
def test_dca_bhtd(Z, H, N_CTX, D_HEAD, chunk_size, local_size, dtype):
    torch.manual_seed(20)

    # triton implementation
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    q_succ = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.2, std=0.2)
    q_inter = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2)
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.5, std=0.2)
    #sm_scale = 1. / math.sqrt(q.shape[-1])

    tri_out, tri_lse = triton_dca_bhtd(
        q, 
        q_succ,
        q_inter,
        k, 
        v, 
        causal=True,
        sm_scale=None,
        dropout_p=0.,
        chunk_len=chunk_size-local_size,
    )   
    
    q, q_succ, q_inter, k, v = [
        rearrange(t, "b h s d -> (b s) h d") for t in [
            q, q_succ, q_inter, k, v
        ]
    ]
    ref_out = _bruteforce_dynamic_chunk_flash_attn_func(
        q,
        q_succ,
        q_inter,
        k,
        v,
        block_table=None,
        softmax_scale=None,
        chunk_size=chunk_size,
        local_size=local_size,
        original_max_position_embeddings=0,
        current_prefill_original_seq_lens_tensor=None,
        k_length=N_CTX,
    )
    ref_out = rearrange(ref_out, '(b s) h d -> b h s d', b=Z)

    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)

