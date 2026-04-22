import torch
import triton
import triton.language as tl

# Reference implementation (slow, but correct)
def reference_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)

# Triton kernel for matrix multiplication
@triton.jit
def matmul_kernel(
    A, B, C,
    stride_a0, stride_a1,
    stride_b0, stride_b1,
    stride_c0, stride_c1,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes C = A @ B where
    A: (M, K), B: (K, N), C: (M, N)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m = m_start + tl.arange(0, BLOCK_M)
    n = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_start = k
        k_idx = k_start + tl.arange(0, BLOCK_K)

        # Load a tile of A
        a_ptr = A + m[:, None] * stride_a0 + k_idx[None, :] * stride_a1
        a_mask = (m[:, None] < M) & (k_idx[None, :] < K)
        a = tl.load(a_ptr, mask=a_mask, other=0.0)

        # Load a tile of B
        b_ptr = B + k_idx[:, None] * stride_b0 + n[None, :] * stride_b1
        b_mask = (k_idx[:, None] < K) & (n[None, :] < N)
        b = tl.load(b_ptr, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Store the result
    c_ptr = C + m[:, None] * stride_c0 + n[None, :] * stride_c1
    c_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(c_ptr, acc, mask=c_mask)

# Wrapper that launches the Triton kernel
def optimized_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes matrix multiplication C = A @ B using a custom Triton kernel.
    Supports arbitrary shapes (M, K) and (K, N) on CUDA devices.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only float32 supported"

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Tile sizes (tuned for typical GPUs)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        A, B, C,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return C

# Example usage (uncomment to run)
# if __name__ == "__main__":
#     torch.manual_seed(0)
#     M, K, N = 512, 512, 512
#     A = torch.randn(M, K, device="cuda", dtype=torch.float32)
#     B = torch.randn(K, N, device="cuda", dtype=torch.float32)
#     C_ref = reference_matmul(A, B)
#     C_opt = optimized_matmul(A, B)
#     print("Error:", torch.allclose(C_ref, C_opt, atol=0.1, rtol=0.1))