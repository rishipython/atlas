import torch
import triton
import triton.language as tl

# Triton kernel that computes softmax along the last dimension of a 2‑D tensor.
# Each program instance processes one row of the input.
@triton.jit
def _softmax_kernel(
    X, Y, M, N,
    stride_x_row, stride_x_col,
    stride_y_row, stride_y_col,
    BLOCK: tl.constexpr,
):
    # Index of the current row
    row = tl.program_id(0)
    if row >= M:
        return

    # ----- 1st pass: compute row‑wise max for numerical stability -----
    max_val = -float("inf")
    for offset in range(0, N, BLOCK):
        col = offset + tl.arange(0, BLOCK)
        mask = col < N
        ptr = X + row * stride_x_row + col * stride_x_col
        # Load with a large negative number for out‑of‑range lanes
        x = tl.load(ptr, mask=mask, other=-float("inf"))
        # Promote to float32 for safety
        x = tl.cast(x, tl.float32)
        max_val = tl.maximum(max_val, tl.max(x, axis=0))

    # ----- 2nd pass: compute sum of exp(x - max) -----
    sum_val = 0.0
    for offset in range(0, N, BLOCK):
        col = offset + tl.arange(0, BLOCK)
        mask = col < N
        ptr = X + row * stride_x_row + col * stride_x_col
        x = tl.load(ptr, mask=mask, other=0.0)
        x = tl.cast(x, tl.float32)
        exp_x = tl.exp(x - max_val)
        sum_val += tl.sum(exp_x, axis=0)

    inv_sum = 1.0 / sum_val

    # ----- 3rd pass: write the softmax values -----
    for offset in range(0, N, BLOCK):
        col = offset + tl.arange(0, BLOCK)
        mask = col < N
        ptr = X + row * stride_x_row + col * stride_x_col
        x = tl.load(ptr, mask=mask, other=0.0)
        x = tl.cast(x, tl.float32)
        exp_x = tl.exp(x - max_val)
        y = exp_x * inv_sum
        out_ptr = Y + row * stride_y_row + col * stride_y_col
        tl.store(out_ptr, y, mask=mask)


def optimized_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax along the last dimension of a 2‑D tensor using a Triton kernel.
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (M, N). Must be 2‑D.
    Returns
    -------
    torch.Tensor
        Softmax of `x` along the last dimension, same shape and dtype as `x`.
    """
    assert x.dim() == 2, "Input must be a 2‑D tensor"
    M, N = x.shape
    # Allocate output tensor with the same dtype
    y = torch.empty_like(x)

    # Strides for row and column access
    stride_x_row = x.stride(0)
    stride_x_col = x.stride(1)
    stride_y_row = y.stride(0)
    stride_y_col = y.stride(1)

    # Launch the kernel: one program instance per row
    BLOCK = 256  # tile size per thread block
    grid = (M,)  # 1‑D grid over rows
    _softmax_kernel[grid](
        x, y, M, N,
        stride_x_row, stride_x_col,
        stride_y_row, stride_y_col,
        BLOCK=BLOCK,
    )
    return y