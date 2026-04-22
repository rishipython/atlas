```python
import torch
import triton
import triton.language as tl

# Triton kernel for matrix multiplication
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_a_m, stride_a_k,
    stride_b_k, stride_b_n,
    stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    row = pid // grid_n
    col = pid % grid_n

    # Accumulator for the output tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Pointers to the current A and B tiles
        a_ptr = A + (row * BLOCK_M) * stride_a_m + (k) * stride_a_k
        b_ptr = B + (k) * stride_b_k + (col * BLOCK_N) * stride_b_n

        # Load tiles with masking to handle boundary conditions
        a_mask = (tl.arange(0, BLOCK_M)[:, None] + row * BLOCK_M < M) & \
                 (tl.arange(0, BLOCK_K)[None, :] + k < K)
        b_mask = (tl.arange(0, BLOCK_K)[:, None] + k < K) & \
                 (tl.arange(0, BLOCK_N)[None, :] + col * BLOCK_N < N)

        a = tl.load(
            a_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_a_m +
                 tl.arange(0, BLOCK_K)[None, :] * stride_a_k,
            mask=a_mask,
            other=0.0,
        )
        b = tl.load(
            b_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_b_k +
                 tl.arange(0, BLOCK_N)[None, :] * stride_b_n,
            mask=b_mask,
            other=0.0,
        )

        acc += tl.dot(a, b)

    # Write the output tile
    c_ptr = C + (row * BLOCK_M) * stride_c_m + (col * BLOCK_N) * stride_c_n
    c_mask = (tl.arange(0, BLOCK_M)[:, None] + row * BLOCK_M < M) & \
             (tl.arange(0, BLOCK_N)[None, :] + col * BLOCK_N < N)

    tl.store(
        c_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_c_m +
               tl.arange(0, BLOCK_N)[None, :] * stride_c_n,
        acc,
        mask=c_mask,
    )


def optimized_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B using an optimized Triton kernel.

    Parameters
    ----------
    A : torch.Tensor
        Matrix of shape (M, K) with dtype torch.float32.
    B : torch.Tensor
        Matrix of shape (K, N) with dtype torch.float32.

    Returns
    -------
    C : torch.Tensor
        Resulting matrix of shape (M, N).
    """
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    # Allocate output tensor on the same device
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Block sizes (tuned for typical GPUs)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    # Grid dimensions
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch the kernel
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
    return