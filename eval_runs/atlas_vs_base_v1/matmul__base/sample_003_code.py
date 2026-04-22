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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # grid dimensions
    row = tl.program_id(0)
    col = tl.program_id(1)
    row_start = row * BLOCK_M
    col_start = col * BLOCK_N

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # main K loop
    for k in range(0, K, BLOCK_K):
        # pointers to tiles of A and B
        a_ptr = A + row_start * stride_a_m + k * stride_a_k
        b_ptr = B + k * stride_b_k + col_start * stride_b_n

        # load tiles with masking for boundary conditions
        a = tl.load(
            a_ptr
            + tl.arange(0, BLOCK_M)[:, None] * stride_a_m
            + tl.arange(0, BLOCK_K)[None, :] * stride_a_k,
            mask=(row_start + tl.arange(0, BLOCK_M)[:, None] < M)
            & (k + tl.arange(0, BLOCK_K)[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + tl.arange(0, BLOCK_K)[:, None] * stride_b_k
            + tl.arange(0, BLOCK_N)[None, :] * stride_b_n,
            mask=(k + tl.arange(0, BLOCK_K)[:, None] < K)
            & (col_start + tl.arange(0, BLOCK_N)[None, :] < N),
            other=0.0,
        )

        # accumulate
        acc += tl.dot(a, b)

    # store the result
    c_ptr = C + row_start * stride_c_m + col_start * stride_c_n
    tl.store(
        c_ptr
        + tl.arange(0, BLOCK_M)[:, None] * stride_c_m
        + tl.arange(0, BLOCK_N)[None, :] * stride_c_n,
        acc,
        mask=(row_start + tl.arange(0, BLOCK_M)[:, None] < M)
        & (col_start + tl.arange(0, BLOCK_N)[None, :] < N),
    )

def optimized_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B using an optimized Triton kernel.
    Supports float32 tensors on CUDA.
    """
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    # Output tensor
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    # Block sizes (tune for your GPU)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128

    # Grid dimensions (2D grid)
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C

# ------------------------------------------------------------------
# Optional test harness (will run if this file is executed directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    for shape in [{'M': 512, 'K': 512, 'N': 512}, {'M': 1024, 'K': 1024, 'N': 1024}]:
        M, K, N = shape['M'], shape['K'], shape['N']
        A = torch.randn(M, K, dtype=torch.float32, device="cuda")
        B = torch.randn(K, N, dtype=torch.float32, device="cuda")

        C_ref = torch.matmul(A, B)
        C_opt = optimized_matmul(A, B)

        # Check numerical closeness
        if not torch.allclose(C_ref, C_opt, atol=0.1, rtol=0.1):
            print(f"Mismatch for shape {shape}")
        else:
            print(f"Success for shape {shape}")