import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A: tl.tensor,
    B: tl.tensor,
    C: tl.tensor,
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    stride_am: tl.int32,
    stride_ak: tl.int32,
    stride_bk: tl.int32,
    stride_bn: tl.int32,
    stride_cm: tl.int32,
    stride_cn: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row = pid_m * BLOCK_M
    col = pid_n * BLOCK_N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load a tile from A: shape (BLOCK_M, BLOCK_K)
        a_ptr = A + row * stride_am + k * stride_ak
        a_ptrs = a_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
        a_mask = (row + tl.arange(0, BLOCK_M)[:, None] < M) & (k + tl.arange(0, BLOCK_K)[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load a tile from B: shape (BLOCK_K, BLOCK_N)
        b_ptr = B + k * stride_bk + col * stride_bn
        b_ptrs = b_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_bk + tl.arange(0, BLOCK_N)[None, :] * stride_bn
        b_mask = (k + tl.arange(0, BLOCK_K)[:, None] < K) & (col + tl.arange(0, BLOCK_N)[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Write the result back to C
    c_ptr = C + row * stride_cm + col * stride_cn
    c_ptrs = c_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_cm + tl.arange(0, BLOCK_N)[None, :] * stride_cn
    c_mask = (row + tl.arange(0, BLOCK_M)[:, None] < M) & (col + tl.arange(0, BLOCK_N)[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def optimized_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B for float32 tensors using a Triton kernel.

    Parameters
    ----------
    A : torch.Tensor
        Matrix of shape (M, K) on CUDA.
    B : torch.Tensor
        Matrix of shape (K, N) on CUDA.

    Returns
    -------
    torch.Tensor
        Resulting matrix of shape (M, N).
    """
    if not (A.is_cuda and B.is_cuda):
        raise ValueError("Both input tensors must be on CUDA device.")
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        raise ValueError("Only float32 dtype is supported.")
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError("Inner dimensions must match.")

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

# Example usage (uncomment to test):
# if __name__ == "__main__":
#     torch.manual_seed(0)
#     for shape in [{'M': 512, 'K': 512, 'N': 512}, {'M': 1024, 'K': 1024, 'N': 1024}]:
#         M, K, N = shape['M'], shape['K'], shape['N']
#         A = torch.randn(M, K, device="cuda", dtype=torch.float32)
#         B = torch.randn(K, N, device="cuda", dtype=torch.float32)
#         C_ref = torch.matmul(A, B)
#         C_opt = optimized_matmul(A, B)
#         assert torch.allclose(C_ref, C_opt, atol=0.1, rtol=0.1)
#     print("All tests passed!")