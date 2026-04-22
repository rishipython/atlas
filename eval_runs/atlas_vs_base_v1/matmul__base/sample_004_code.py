import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A: tl.tensor,          # (M, K)
    B: tl.tensor,          # (K, N)
    C: tl.tensor,          # (M, N) output
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
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
    """
    Triton kernel that computes C = A @ B.
    Each program handles a BLOCK_M x BLOCK_N tile of C.
    """
    row = tl.program_id(0) * BLOCK_M
    col = tl.program_id(1) * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load a tile of A: (BLOCK_M, BLOCK_K)
        a_ptr = A + (row + tl.arange(0, BLOCK_M)) * stride_a_m \
                      + (k + tl.arange(0, BLOCK_K)) * stride_a_k
        # Load a tile of B: (BLOCK_K, BLOCK_N)
        b_ptr = B + (k + tl.arange(0, BLOCK_K)) * stride_b_k \
                      + (col + tl.arange(0, BLOCK_N)) * stride_b_n

        a = tl.load(a_ptr)
        b = tl.load(b_ptr)

        acc += tl.dot(a, b)

    # Write result tile to C, masking out-of-range elements
    mask_row = row + tl.arange(0, BLOCK_M) < M
    mask_col = col + tl.arange(0, BLOCK_N) < N
    mask = mask_row[:, None] & mask_col[None, :]

    c_ptr = C + (row + tl.arange(0, BLOCK_M)) * stride_c_m \
                + (col + tl.arange(0, BLOCK_N)) * stride_c_n
    tl.store(c_ptr, acc, mask=mask)


def optimized_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B using a Triton kernel.
    Supports 2-D float32 tensors; M, K, N are arbitrary.
    """
    assert A.ndim == 2 and B.ndim == 2
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match."
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    assert A.device == B.device

    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

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

# Example usage and sanity check
if __name__ == "__main__":
    torch.manual_seed(0)
    for shape in [{'M': 512, 'K': 512, 'N': 512}, {'M': 1024, 'K': 1024, 'N': 1024}]:
        M, K, N = shape['M'], shape['K'], shape['N']
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)
        C_ref = torch.matmul(A, B)
        C_opt = optimized_matmul(A, B)
        print(f"Shape {shape}: diff = {torch.max(torch.abs(C_ref - C_opt))}")
        assert torch.allclose(C_ref, C_opt, atol=0.1, rtol=0.1)
    print("All tests passed.")