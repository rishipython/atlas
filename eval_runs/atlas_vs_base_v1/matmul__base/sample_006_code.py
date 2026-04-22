import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr: tl.tensor,
    B_ptr: tl.tensor,
    C_ptr: tl.tensor,
    M: tl.int64,
    N: tl.int64,
    K: tl.int64,
    stride_a_m: tl.int64,
    stride_a_k: tl.int64,
    stride_b_k: tl.int64,
    stride_b_n: tl.int64,
    stride_c_m: tl.int64,
    stride_c_n: tl.int64,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # block indices
    block_row = tl.program_id(axis=0)
    block_col = tl.program_id(axis=1)

    row_start = block_row * BLOCK_M
    col_start = block_col * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over K dimension
    for k in range(0, K, BLOCK_K):
        k_start = k

        # indices for A block
        row_a = row_start + tl.arange(0, BLOCK_M)[:, None]
        col_a = k_start + tl.arange(0, BLOCK_K)[None, :]
        mask_a = (row_a < M) & (col_a < K)

        a_ptr = A_ptr + row_a * stride_a_m + col_a * stride_a_k
        a = tl.load(a_ptr, mask=mask_a, other=0.0)

        # indices for B block
        row_b = k_start + tl.arange(0, BLOCK_K)[:, None]
        col_b = col_start + tl.arange(0, BLOCK_N)[None, :]
        mask_b = (row_b < K) & (col_b < N)

        b_ptr = B_ptr + row_b * stride_b_k + col_b * stride_b_n
        b = tl.load(b_ptr, mask=mask_b, other=0.0)

        acc += tl.dot(a, b)

    # write back to C
    row_c = row_start + tl.arange(0, BLOCK_M)[:, None]
    col_c = col_start + tl.arange(0, BLOCK_N)[None, :]
    mask_c = (row_c < M) & (col_c < N)

    c_ptr = C_ptr + row_c * stride_c_m + col_c * stride_c_n
    tl.store(c_ptr, acc, mask=mask_c)

def optimized_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B for 2-D float32 tensors using a Triton kernel.
    """
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    assert A.ndim == 2 and B.ndim == 2
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match."

    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C