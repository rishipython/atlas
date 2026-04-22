```python
import torch
import triton
import triton.language as tl

# Triton kernel for LayerNorm over the last dimension of a 2‑D tensor.
@triton.jit
def _layernorm_kernel(
    x_ptr: tl.tensor,
    w_ptr: tl.tensor,
    b_ptr: tl.tensor,
    y_ptr: tl.tensor,
    M: tl.int32,
    N: tl.int32,
    stride_row: tl.int32,
    stride_col: tl.int32,
    stride_w: tl.int32,
    stride_b: tl.int32,
    eps: tl.float32,
    BLOCK_N: tl.constexpr,
):
    # Each program id processes one row.
    row = tl.program_id(0)
    if row >= M:
        return

    # First pass: compute mean and variance.
    sum_ = 0.0
    sum_sq = 0.0
    start = 0
    col_idx = tl.arange(0, BLOCK_N)

    while start < N:
        # Compute the linear offset for the current chunk.
        offset = row * stride_row + start
        mask = (start + col_idx) < N

        # Load the chunk, cast to float32 for accumulation.
        x = tl.load(x_ptr + offset + col_idx * stride_col, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)

        # Accumulate sum and sum of squares.
        sum_ += tl.sum(x_f32, axis=0)
        sum_sq += tl.sum(x_f32 * x_f32, axis=0)

        start += BLOCK_N

    mean = sum_ / tl.float32(N)
    var = sum_sq / tl.float32(N) - mean * mean
    denom = tl.math.rsqrt(var + eps)

    # Second pass: compute normalized output.
    start = 0
    while start < N:
        offset = row * stride_row + start
        mask = (start + col_idx) < N

        x = tl.load(x_ptr + offset + col_idx * stride_col, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)

        # Normalized value.
        norm = (x_f32 - mean) * denom

        # Load weight and bias (they are 1‑D).
        w = tl.load(w_ptr + start + col_idx * stride_w, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + start + col_idx * stride_b, mask=mask, other=0.0).to(tl.float32)

        y = norm * w + b
        tl.store(y_ptr + offset + col_idx * stride_col, y, mask=mask)


def optimized_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Layer Normalization over the last dimension of a 2‑D tensor using a Triton kernel.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (M, N).
    weight : torch.Tensor
        Learnable scale parameters of shape (N,).
    bias : torch.Tensor
        Learnable bias parameters of shape (N,).
    eps : float, optional
        Small value added for numerical stability (default 1e-5).

    Returns
    -------
    torch.Tensor
        Normalized tensor of the same shape and dtype as ``x``.
    """
    assert x.dim() == 2, "Input must be 2‑D"
    assert weight.shape[0] == x.shape[1], "Weight length must match last dimension"
    assert bias.shape[0] == x.shape[1], "Bias length must match last dimension"

    M, N = x.shape
    y = torch.empty_like(x)

    # Triton constants
    BLOCK_N = 128  # Tunable block size for the inner dimension

    # Grid: one program per row
    grid = (triton.cdiv(M