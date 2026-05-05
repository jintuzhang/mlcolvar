import math
import warnings

import torch

__all__ = ['TriAttentionFunction', 'tri_attention_run', 'TRITON_AVAILABLE']


try:  # pragma: no cover - optional dependency
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except (
    ImportError,
    RuntimeError,
) as triton_error:  # pragma: no cover - import fallback
    triton = None  # type: ignore
    tl = None  # type: ignore
    TRITON_AVAILABLE = False
    _TRITON_ERROR = triton_error


def get_tag(x: int) -> int:
    if x > 4096:
        return 4096
    return math.ceil(x / 32) * 32


QK_SCALE = 1.4426950408889634

FWD_META = {'BLOCK_M': 16, 'BLOCK_N': 32, 'num_warps': 1, 'num_stages': 2}
DKDV_META = {'BLOCK_M': 32, 'BLOCK_N': 32, 'num_warps': 1, 'num_stages': 2}
DQ_META = {'BLOCK_M': 16, 'BLOCK_N': 32, 'num_warps': 2, 'num_stages': 1}
DBIAS2_META = {'BLOCK_M': 16, 'BLOCK_N': 16, 'num_warps': 2, 'num_stages': 2}
PREPROCESS_META = {'BLOCK_M': 16, 'num_warps': 2, 'num_stages': 2}


if TRITON_AVAILABLE:

    @triton.jit
    def _attention_fwd(
        Q,
        K,
        V,
        O,
        Bias1,
        Bias2,
        M,
        N,
        S,
        H,
        HEAD_DIM: tl.constexpr,
        TAG_N: tl.constexpr,
        deterministic: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        input_dtype = Q.dtype.element_ty

        bn = tl.program_id(2)
        h = tl.program_id(1)
        s = tl.program_id(0) * BLOCK_M
        b = bn // N
        n = bn % N

        Bs = N * S * H * HEAD_DIM
        Ns = S * H * HEAD_DIM
        Ss = H * HEAD_DIM
        Hs = HEAD_DIM

        qkvo_offset = b * Bs + n * Ns + h * Hs
        bias1_offset = b * N * S + n * S
        bias2_offset = b * S * S * H + h * S * S
        m_offset = b * N * S * H + n * S * H + h

        q_ptr = tl.make_block_ptr(
            base=Q + qkvo_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        k_ptr = tl.make_block_ptr(
            base=K + qkvo_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(0, 1),
        )

        v_ptr = tl.make_block_ptr(
            base=V + qkvo_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )

        o_ptr = tl.make_block_ptr(
            base=O + qkvo_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        bias1_ptr = tl.make_block_ptr(
            base=Bias1 + bias1_offset,
            shape=(S, S),
            strides=(0, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )

        bias2_ptr = tl.make_block_ptr(
            base=Bias2 + bias2_offset,
            shape=(S, S),
            strides=(S, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )

        m_ptr = tl.make_block_ptr(
            base=M + m_offset,
            shape=(S, 1),
            strides=(H, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, 1),
            order=(0, 1),
        )

        qk_scale = 1.4426950408889634

        m_i = tl.zeros([BLOCK_M], dtype=input_dtype) - float('inf')
        l_i = tl.zeros([BLOCK_M], dtype=input_dtype) + 1
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=input_dtype)

        q = tl.load(q_ptr, boundary_check=(0, 1), padding_option='zero')

        offset_q = (s + tl.arange(0, BLOCK_M))[:, None] < S
        offset_k = tl.arange(0, BLOCK_N)[None, :]
        for _ in range(tl.cdiv(S, BLOCK_N)):
            k = tl.load(k_ptr, boundary_check=(0, 1), padding_option='zero')
            bias1 = tl.load(
                bias1_ptr, boundary_check=(0, 1), padding_option='zero'
            )
            bias2 = tl.load(
                bias2_ptr, boundary_check=(0, 1), padding_option='zero'
            )
            mask = offset_q * (offset_k < S)
            qk = bias1.to(input_dtype) + bias2.to(input_dtype)
            qk = (
                tl.dot(
                    q,
                    tl.trans(k),
                    qk,
                    input_precision='ieee' if deterministic else 'tf32',
                    out_dtype=input_dtype,
                )
                * qk_scale
            )
            qk = tl.where(mask, qk, -1e6)
            m_ij = tl.maximum(tl.max(qk, 1), m_i)
            qk = qk - m_ij[:, None]
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(qk)
            l_ij = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v = tl.load(v_ptr, boundary_check=(0, 1), padding_option='zero')
            p = p.to(input_dtype)
            acc = tl.dot(
                p,
                v,
                acc,
                input_precision='ieee' if deterministic else 'tf32',
                out_dtype=input_dtype,
            )

            l_i = l_ij
            m_i = m_ij

            k_ptr = tl.advance(k_ptr, (BLOCK_N, 0))
            v_ptr = tl.advance(v_ptr, (BLOCK_N, 0))
            bias1_ptr = tl.advance(bias1_ptr, (0, BLOCK_N))
            bias2_ptr = tl.advance(bias2_ptr, (0, BLOCK_N))
            offset_k += BLOCK_N

        acc = acc / l_i[:, None]
        m_i += tl.math.log2(l_i)
        acc = acc.to(input_dtype)
        tl.store(o_ptr, acc, boundary_check=(0, 1))
        tl.store(m_ptr, m_i[:, None], boundary_check=(0, 1))

    @triton.jit
    def _attention_bwd_preprocess(
        O,
        DO,
        Delta,
        N,
        S,
        H,
        HEAD_DIM: tl.constexpr,
        TAG_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        input_dtype = O.dtype.element_ty

        bn = tl.program_id(2)
        h = tl.program_id(1)
        s = tl.program_id(0) * BLOCK_M
        b = bn // N
        n = bn % N

        Bs = N * S * H * HEAD_DIM
        Ns = S * H * HEAD_DIM
        Ss = H * HEAD_DIM
        Hs = HEAD_DIM

        o_offset = b * Bs + n * Ns + h * Hs
        delta_offset = b * N * H * S + n * S * H + h

        o_ptr = tl.make_block_ptr(
            base=O + o_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(0, 1),
        )

        do_ptr = tl.make_block_ptr(
            base=DO + o_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(0, 1),
        )

        delta_ptr = tl.make_block_ptr(
            base=Delta + delta_offset,
            shape=(S, 1),
            strides=(H, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, 1),
            order=(0, 1),
        )

        o = tl.load(o_ptr, boundary_check=(0, 1), padding_option='zero')
        do = tl.load(do_ptr, boundary_check=(0, 1), padding_option='zero')
        delta = tl.sum(o.to(input_dtype) * do.to(input_dtype), axis=1)
        tl.store(delta_ptr, delta[:, None], boundary_check=(0, 1))

    @triton.jit
    def _attention_bwd_dkdv(
        Q,
        K,
        V,
        Bias1,
        Bias2,
        M,
        Delta,
        DK,
        DV,
        DO,
        N,
        S,
        H,
        HEAD_DIM: tl.constexpr,
        TAG_N: tl.constexpr,
        deterministic: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        input_dtype = Q.dtype.element_ty

        bn = tl.program_id(2)
        h = tl.program_id(1)
        s = tl.program_id(0) * BLOCK_M
        b = bn // N
        n = bn % N

        Bs = N * S * H * HEAD_DIM
        Ns = S * H * HEAD_DIM
        Ss = H * HEAD_DIM
        Hs = HEAD_DIM

        qkvo_offset = b * Bs + n * Ns + h * Hs
        bias1_offset = b * N * S + n * S
        bias2_offset = b * S * S * H + h * S * S
        delta_m_offset = b * N * H * S + n * S * H + h

        q_ptr = tl.make_block_ptr(
            base=Q + qkvo_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )

        k_ptr = tl.make_block_ptr(
            base=K + qkvo_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(0, 1),
        )

        dk_ptr = tl.make_block_ptr(
            base=DK + qkvo_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(0, 1),
        )

        v_ptr = tl.make_block_ptr(
            base=V + qkvo_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(0, 1),
        )

        dv_ptr = tl.make_block_ptr(
            base=DV + qkvo_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(0, 1),
        )

        do_ptr = tl.make_block_ptr(
            base=DO + qkvo_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(0, 1),
        )

        bias1_ptr = tl.make_block_ptr(
            base=Bias1 + bias1_offset,
            shape=(S, S),
            strides=(0, 1),
            offsets=(0, s),
            block_shape=(BLOCK_N, BLOCK_M),
            order=(1, 0),
        )

        bias2_ptr = tl.make_block_ptr(
            base=Bias2 + bias2_offset,
            shape=(S, S),
            strides=(S, 1),
            offsets=(0, s),
            block_shape=(BLOCK_N, BLOCK_M),
            order=(1, 0),
        )

        m_ptr = tl.make_block_ptr(
            base=M + delta_m_offset,
            shape=(S, 1),
            strides=(H, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, 1),
            order=(0, 1),
        )

        delta_ptr = tl.make_block_ptr(
            base=Delta + delta_m_offset,
            shape=(S, 1),
            strides=(H, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, 1),
            order=(0, 1),
        )

        k = tl.load(k_ptr, boundary_check=(0, 1), padding_option='zero')
        v = tl.load(v_ptr, boundary_check=(0, 1), padding_option='zero')
        dv = tl.zeros([BLOCK_M, HEAD_DIM], dtype=input_dtype)
        dk = tl.zeros([BLOCK_M, HEAD_DIM], dtype=input_dtype)
        qk_scale = 1.4426950408889634

        for _ in range(tl.cdiv(S, BLOCK_N)):
            q = tl.load(q_ptr, boundary_check=(0, 1), padding_option='zero')
            bias1 = tl.load(
                bias1_ptr, boundary_check=(0, 1), padding_option='zero'
            )
            bias2 = tl.load(
                bias2_ptr, boundary_check=(0, 1), padding_option='zero'
            )
            m = tl.load(m_ptr, boundary_check=(0, 1), padding_option='zero')
            qk = bias1.to(input_dtype) + bias2.to(input_dtype)
            qk = (
                tl.dot(
                    q,
                    tl.trans(k),
                    qk,
                    input_precision='ieee' if deterministic else 'tf32',
                    out_dtype=input_dtype,
                )
            ) * qk_scale
            p = tl.math.exp2(qk - m)
            do = tl.load(do_ptr, boundary_check=(0, 1), padding_option='zero')
            dv = tl.dot(
                tl.trans(p.to(input_dtype)),
                do,
                dv,
                input_precision='ieee' if deterministic else 'tf32',
                out_dtype=input_dtype,
            )

            dp = tl.dot(
                do,
                tl.trans(v),
                input_precision='ieee' if deterministic else 'tf32',
                out_dtype=input_dtype,
            )
            delta = tl.load(
                delta_ptr, boundary_check=(0, 1), padding_option='zero'
            )
            dqk = p * (dp - delta)
            dk = tl.dot(
                tl.trans(dqk).to(input_dtype),
                q,
                dk,
                input_precision='ieee' if deterministic else 'tf32',
                out_dtype=input_dtype,
            )

            q_ptr = tl.advance(q_ptr, (BLOCK_N, 0))
            do_ptr = tl.advance(do_ptr, (BLOCK_N, 0))
            bias1_ptr = tl.advance(bias1_ptr, (BLOCK_N, 0))
            bias2_ptr = tl.advance(bias2_ptr, (BLOCK_N, 0))
            m_ptr = tl.advance(m_ptr, (BLOCK_N, 0))
            delta_ptr = tl.advance(delta_ptr, (BLOCK_N, 0))
        tl.store(dv_ptr, dv.to(input_dtype), boundary_check=(0, 1))
        tl.store(dk_ptr, dk.to(input_dtype), boundary_check=(0, 1))

    @triton.jit
    def _attention_bwd_dq(
        Q,
        K,
        V,
        Bias1,
        Bias2,
        M,
        Delta,
        DQ,
        DO,
        N,
        S,
        H,
        HEAD_DIM: tl.constexpr,
        TAG_N: tl.constexpr,
        deterministic: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        input_dtype = Q.dtype.element_ty

        bn = tl.program_id(2)
        h = tl.program_id(1)
        s = tl.program_id(0) * BLOCK_M
        b = bn // N
        n = bn % N

        Bs = N * S * H * HEAD_DIM
        Ns = S * H * HEAD_DIM
        Ss = H * HEAD_DIM
        Hs = HEAD_DIM

        qkvo_offset = b * Bs + n * Ns + h * Hs
        bias1_offset = b * N * S + n * S
        bias2_offset = b * S * S * H + h * S * S
        delta_m_offset = b * N * H * S + n * S * H + h

        q_ptr = tl.make_block_ptr(
            base=Q + qkvo_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        dq_ptr = tl.make_block_ptr(
            base=DQ + qkvo_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        k_ptr = tl.make_block_ptr(
            base=K + qkvo_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(0, 1),
        )

        v_ptr = tl.make_block_ptr(
            base=V + qkvo_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(0, 1),
        )

        do_ptr = tl.make_block_ptr(
            base=DO + qkvo_offset,
            shape=(S, HEAD_DIM),
            strides=(Ss, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(0, 1),
        )

        bias1_ptr = tl.make_block_ptr(
            base=Bias1 + bias1_offset,
            shape=(S, S),
            strides=(0, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )

        bias2_ptr = tl.make_block_ptr(
            base=Bias2 + bias2_offset,
            shape=(S, S),
            strides=(S, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )

        m_ptr = tl.make_block_ptr(
            base=M + delta_m_offset,
            shape=(S, 1),
            strides=(H, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, 1),
            order=(0, 1),
        )

        delta_ptr = tl.make_block_ptr(
            base=Delta + delta_m_offset,
            shape=(S, 1),
            strides=(H, 1),
            offsets=(s, 0),
            block_shape=(BLOCK_M, 1),
            order=(0, 1),
        )

        q = tl.load(q_ptr, boundary_check=(0, 1), padding_option='zero')
        do = tl.load(do_ptr, boundary_check=(0, 1), padding_option='zero')
        delta = tl.load(
            delta_ptr, boundary_check=(0, 1), padding_option='zero'
        )
        m = tl.load(m_ptr, boundary_check=(0, 1), padding_option='zero')
        dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=input_dtype)
        qk_scale = 1.4426950408889634
        for _ in range(tl.cdiv(S, BLOCK_N)):
            k = tl.load(k_ptr, boundary_check=(0, 1), padding_option='zero')
            bias1 = tl.load(
                bias1_ptr, boundary_check=(0, 1), padding_option='zero'
            )
            bias2 = tl.load(
                bias2_ptr, boundary_check=(0, 1), padding_option='zero'
            )

            qk = bias1.to(input_dtype) + bias2.to(input_dtype)
            qk = (
                tl.dot(
                    q,
                    tl.trans(k),
                    qk,
                    input_precision='ieee' if deterministic else 'tf32',
                    out_dtype=input_dtype,
                )
                * qk_scale
            )
            p = tl.math.exp2(qk - m)
            v = tl.load(v_ptr, boundary_check=(0, 1), padding_option='zero')
            dp = tl.dot(
                do,
                tl.trans(v),
                input_precision='ieee' if deterministic else 'tf32',
                out_dtype=input_dtype,
            )

            dqk = p * (dp - delta)
            dq = tl.dot(
                dqk.to(input_dtype),
                k,
                dq,
                input_precision='ieee' if deterministic else 'tf32',
                out_dtype=input_dtype,
            )

            k_ptr = tl.advance(k_ptr, (BLOCK_N, 0))
            v_ptr = tl.advance(v_ptr, (BLOCK_N, 0))
            bias1_ptr = tl.advance(bias1_ptr, (0, BLOCK_N))
            bias2_ptr = tl.advance(bias2_ptr, (0, BLOCK_N))
        tl.store(dq_ptr, dq.to(input_dtype), boundary_check=(0, 1))

    @triton.jit
    def _attention_bwd_dbias2(
        Q,
        K,
        V,
        Bias1,
        Bias2,
        M,
        Delta,
        DBias2,
        DO,
        N,
        S,
        H,
        HEAD_DIM: tl.constexpr,
        TAG_N: tl.constexpr,
        deterministic: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        input_dtype = Q.dtype.element_ty

        s1 = tl.program_id(0) * BLOCK_M
        s2 = tl.program_id(1) * BLOCK_N
        bh = tl.program_id(2)
        b = bh // H
        h = bh % H

        Bs = N * S * H * HEAD_DIM
        Ns = S * H * HEAD_DIM
        Ss = H * HEAD_DIM
        Hs = HEAD_DIM

        bias1_offset = b * N * S
        bias2_offset = b * S * S * H + h * S * S

        qkvo_offset = b * Bs + h * Hs

        delta_m_offset = b * N * H * S + h

        bias2_ptr = tl.make_block_ptr(
            base=Bias2 + bias2_offset,
            shape=(S, S),
            strides=(S, 1),
            offsets=(s1, s2),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )

        dbias2_ptr = tl.make_block_ptr(
            base=DBias2 + bias2_offset,
            shape=(S, S),
            strides=(S, 1),
            offsets=(s1, s2),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )

        bias2 = tl.load(
            bias2_ptr, boundary_check=(0, 1), padding_option='zero'
        )
        dbias2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=input_dtype)
        qk_scale = 1.4426950408889634
        for n in range(0, N, 1):
            q_ptr = tl.make_block_ptr(
                base=Q + qkvo_offset + n * Ns,
                shape=(S, HEAD_DIM),
                strides=(Ss, 1),
                offsets=(s1, 0),
                block_shape=(BLOCK_M, HEAD_DIM),
                order=(1, 0),
            )

            k_ptr = tl.make_block_ptr(
                base=K + qkvo_offset + n * Ns,
                shape=(S, HEAD_DIM),
                strides=(Ss, 1),
                offsets=(s2, 0),
                block_shape=(BLOCK_N, HEAD_DIM),
                order=(1, 0),
            )

            v_ptr = tl.make_block_ptr(
                base=V + qkvo_offset + n * Ns,
                shape=(S, HEAD_DIM),
                strides=(Ss, 1),
                offsets=(s2, 0),
                block_shape=(BLOCK_N, HEAD_DIM),
                order=(1, 0),
            )

            do_ptr = tl.make_block_ptr(
                base=DO + qkvo_offset + n * Ns,
                shape=(S, HEAD_DIM),
                strides=(Ss, 1),
                offsets=(s1, 0),
                block_shape=(BLOCK_M, HEAD_DIM),
                order=(1, 0),
            )

            bias1_ptr = tl.make_block_ptr(
                base=Bias1 + bias1_offset + n * S,
                shape=(S, S),
                strides=(0, 1),
                offsets=(s1, s2),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )

            m_ptr = tl.make_block_ptr(
                base=M + delta_m_offset + n * S * H,
                shape=(S, 1),
                strides=(H, 1),
                offsets=(s1, 0),
                block_shape=(BLOCK_M, 1),
                order=(0, 1),
            )

            delta_ptr = tl.make_block_ptr(
                base=Delta + delta_m_offset + n * S * H,
                shape=(S, 1),
                strides=(H, 1),
                offsets=(s1, 0),
                block_shape=(BLOCK_M, 1),
                order=(0, 1),
            )

            q = tl.load(q_ptr, boundary_check=(0, 1), padding_option='zero')
            k = tl.load(k_ptr, boundary_check=(0, 1), padding_option='zero')
            v = tl.load(v_ptr, boundary_check=(0, 1), padding_option='zero')
            do = tl.load(do_ptr, boundary_check=(0, 1), padding_option='zero')
            delta = tl.load(
                delta_ptr, boundary_check=(0, 1), padding_option='zero'
            )
            m = tl.load(m_ptr, boundary_check=(0, 1), padding_option='zero')
            bias1 = tl.load(
                bias1_ptr, boundary_check=(0, 1), padding_option='zero'
            )

            qk = bias1.to(input_dtype) + bias2.to(input_dtype)
            qk = (
                tl.dot(
                    q,
                    tl.trans(k),
                    qk,
                    input_precision='ieee' if deterministic else 'tf32',
                    out_dtype=input_dtype,
                )
                * qk_scale
            )
            p = tl.math.exp2(qk - m)
            v = tl.load(v_ptr, boundary_check=(0, 1), padding_option='zero')
            dp = tl.dot(
                do,
                tl.trans(v),
                input_precision='ieee' if deterministic else 'tf32',
                out_dtype=input_dtype,
            )

            dqk = p * (dp - delta)
            dbias2 += dqk

        tl.store(dbias2_ptr, dbias2.to(input_dtype), boundary_check=(0, 1))

    class TriAttentionFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Q, K, V, Bias1, Bias2, deterministic: bool = False):
            input_dtype = Q.dtype

            Q = Q.contiguous()
            K = K.contiguous()
            V = V.contiguous()
            Bias1 = Bias1.contiguous()
            Bias2 = Bias2.contiguous()

            B, N, S, H, D = Q.shape
            TAG_N = get_tag(S)
            O = torch.empty_like(Q)
            M = torch.empty((B, N, S, H), device=Q.device, dtype=input_dtype)

            grid = (
                triton.cdiv(S, FWD_META['BLOCK_M']),
                H,
                B * N,
            )
            _attention_fwd[grid](
                Q,
                K,
                V,
                O,
                Bias1,
                Bias2,
                M,
                N,
                S,
                H,
                HEAD_DIM=D,
                TAG_N=TAG_N,
                deterministic=deterministic,
                BLOCK_M=FWD_META['BLOCK_M'],
                BLOCK_N=FWD_META['BLOCK_N'],
                num_warps=FWD_META['num_warps'],
                num_stages=FWD_META['num_stages'],
            )

            ctx.save_for_backward(Q, K, V, Bias1, Bias2, O, M)
            ctx.deterministic = deterministic
            return O

        @staticmethod
        def backward(ctx, DO):
            Q, K, V, Bias1, Bias2, O, M = ctx.saved_tensors
            deterministic = ctx.deterministic
            B, N, S, H, D = Q.shape
            TAG_N = get_tag(S)
            DQ = torch.empty_like(Q)
            DK = torch.empty_like(K)
            DV = torch.empty_like(V)
            DBias1 = torch.empty_like(Bias1)
            DBias2 = torch.empty_like(Bias2)
            Delta = torch.empty_like(M)

            grid = (
                triton.cdiv(S, PREPROCESS_META['BLOCK_M']),
                H,
                B * N,
            )
            _attention_bwd_preprocess[grid](
                O,
                DO,
                Delta,
                N,
                S,
                H,
                HEAD_DIM=D,
                TAG_N=TAG_N,
                BLOCK_M=PREPROCESS_META['BLOCK_M'],
                num_warps=PREPROCESS_META['num_warps'],
                num_stages=PREPROCESS_META['num_stages'],
            )

            grid = (
                triton.cdiv(S, DKDV_META['BLOCK_M']),
                H,
                B * N,
            )

            _attention_bwd_dkdv[grid](
                Q,
                K,
                V,
                Bias1,
                Bias2,
                M,
                Delta,
                DK,
                DV,
                DO,
                N,
                S,
                H,
                HEAD_DIM=D,
                TAG_N=TAG_N,
                deterministic=deterministic,
                BLOCK_M=DKDV_META['BLOCK_M'],
                BLOCK_N=DKDV_META['BLOCK_N'],
                num_warps=DKDV_META['num_warps'],
                num_stages=DKDV_META['num_stages'],
            )

            grid = (
                triton.cdiv(S, DQ_META['BLOCK_M']),
                H,
                B * N,
            )
            _attention_bwd_dq[grid](
                Q,
                K,
                V,
                Bias1,
                Bias2,
                M,
                Delta,
                DQ,
                DO,
                N,
                S,
                H,
                HEAD_DIM=D,
                TAG_N=TAG_N,
                deterministic=deterministic,
                BLOCK_M=DQ_META['BLOCK_M'],
                BLOCK_N=DQ_META['BLOCK_N'],
                num_warps=DQ_META['num_warps'],
                num_stages=DQ_META['num_stages'],
            )

            grid = (
                triton.cdiv(S, DBIAS2_META['BLOCK_M']),
                triton.cdiv(S, DBIAS2_META['BLOCK_N']),
                B * H,
            )
            _attention_bwd_dbias2[grid](
                Q,
                K,
                V,
                Bias1,
                Bias2,
                M,
                Delta,
                DBias2,
                DO,
                N,
                S,
                H,
                HEAD_DIM=D,
                TAG_N=TAG_N,
                deterministic=deterministic,
                BLOCK_M=DBIAS2_META['BLOCK_M'],
                BLOCK_N=DBIAS2_META['BLOCK_N'],
                num_warps=DBIAS2_META['num_warps'],
                num_stages=DBIAS2_META['num_stages'],
            )
            return DQ, DK, DV, DBias1, DBias2, None

else:

    class TriAttentionFunction:
        """Fallback implementation using PyTorch attention when Triton is unavailable."""

        @staticmethod
        def apply(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            bias1: torch.Tensor,
            bias2: torch.Tensor,
            deterministic: bool = False,
        ) -> torch.Tensor:
            B, N, S, H, D = q.shape

            q_t = q.permute(0, 1, 3, 2, 4).reshape(B * N * H, S, D)
            k_t = k.permute(0, 1, 3, 2, 4).reshape(B * N * H, S, D)
            v_t = v.permute(0, 1, 3, 2, 4).reshape(B * N * H, S, D)

            logits = torch.matmul(q_t, k_t.transpose(-1, -2))

            if bias1 is not None:
                b1 = bias1.reshape(B, N, *bias1.shape[-3:])
                b1 = b1.expand(-1, -1, S, -1, -1)
                b1 = b1.reshape(B * N, S, -1, bias1.shape[-1])
                b1 = b1.squeeze(-3).squeeze(-2)
                logits += (
                    b1.unsqueeze(1)
                    .expand(-1, H, -1, -1)
                    .reshape(B * N * H, S, S)
                )

            if bias2 is not None:
                b2 = bias2.reshape(B, N, *bias2.shape[-3:])
                b2 = b2.expand(-1, -1, -1, S, -1)
                b2 = b2.reshape(B * N, -1, S, S).squeeze(-3)
                logits += (
                    b2.unsqueeze(1)
                    .expand(-1, H, -1, -1)
                    .reshape(B * N * H, S, S)
                )

            attn = torch.softmax(logits, dim=-1)
            out = attn @ v_t
            out = out.reshape(B, N, H, S, D).permute(0, 1, 3, 2, 4)
            warnings.warn(
                f'Triton is unavailable ({_TRITON_ERROR}). Falling back to PyTorch attention.',
                UserWarning,
                stacklevel=2,
            )
            return out


def tri_attention_run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias1: torch.Tensor,
    bias2: torch.Tensor,
    deterministic: bool = True,
) -> torch.Tensor:
    return TriAttentionFunction.apply(q, k, v, bias1, bias2, deterministic)
