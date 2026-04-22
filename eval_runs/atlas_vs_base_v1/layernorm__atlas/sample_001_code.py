import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm_kernel(
    x_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    y_ptr: tl.tensor,
    N: tl.int32,
    stride_x: tl.int32,
    eps: tl.float32,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel that performs layer normalization over the last dimension
    of a 2‑D tensor. Each program processes one row of the input.
    """
    row = tl.program_id(0)

    # ---------- First pass: compute mean and variance ----------
    sum_   = tl.zeros([1], dtype=tl.float32)[0]
    sum_sq = tl.zeros([1], dtype=tl.float32)[0]

    for offset in range(0, N, BLOCK_N):
        offs = offset + tl.arange(0, BLOCK_N)
        mask = offs < N

        ptr = x_ptr + row * stride_x + offs
        x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)

        sum_   += tl.sum(x, mask=mask)
        sum_sq += tl.sum(x * x, mask=mask)

    mean = sum_ / N
    var  = sum_sq / N - mean * mean
    denom = tl.sqrt(var + eps)

    # ---------- Second pass: normalize and apply weight/bias ----------
    for offset in range(0, N, BLOCK_N):
        offs = offset + tl.arange(0, BLOCK_N)
        mask = offs < N

        ptr_x = x_ptr + row * stride_x + offs
        ptr_w = weight_ptr + offs
        ptr_b = bias_ptr + offs

        x = tl.load(ptr_x, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ptr_w, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(ptr_b, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) / denom * w + b

        out_ptr = y_ptr + row * stride_x + offs
        tl.store(out_ptr, y, mask=mask)


def optimized_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Compute layer normalization over the last dimension of a 2‑D tensor using a Triton kernel.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (M, N).
    weight : torch.Tensor
        Weight vector of shape (N,).
    bias : torch.Tensor
        Bias vector of shape (N,).
    eps : float, optional
        Small epsilon for numerical stability (default: 1e-5).

    Returns
    -------
    torch.Tensor
        Normalized tensor of shape (M, N).
    """
    assert x.dim() == 2, "Input must be 2‑D"
    M, N = x.shape
    assert weight.shape[0] == N, "Weight length must match last dimension"
    assert bias.shape[0] == N, "Bias length must match last dimension"

    # Allocate output tensor on the same device and dtype as input
    y = torch.empty_like(x)

    # Triton grid: one program per row
    grid = (M,)

    # Choose a block size that is a multiple of 8 for efficient memory access
    BLOCK_N = 128

    _layernorm_kernel[grid](
        x,
        weight,
        bias,
        y,
        N,
        x.stride(0),
        eps,
        BLOCK_N=BLOCK_N,
    )

    return y