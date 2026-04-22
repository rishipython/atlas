```python
import torch
import triton
import triton.language as tl

# Kernel to compute the row‑wise maximum
@triton.jit
def _softmax_row_max(
    X_ptr, max_ptr, N, stride_x,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X_ptr + row * stride_x
    max_val = -float("inf")
    i = 0
    while i < N:
        offset = i
        mask = (offset + tl.arange(0, BLOCK_N)) < N
        ptr = x_ptr + offset * stride_x
        x = tl.load(ptr, mask=mask, other=-float("inf"))
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
        i += BLOCK_N
    tl.store(max_ptr + row, max_val)

# Kernel to compute exp(x - max) and the row‑wise sum of exponentials
@triton.jit
def _softmax_row_exp_sum(
    X_ptr, Y_ptr, max_ptr, sum_ptr, N, stride_x, stride_y,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X_ptr + row * stride_x
    y_ptr = Y_ptr + row * stride_y
    max_val = tl.load(max_ptr + row)
    exp_sum = 0.0
    i = 0
    while i < N:
        offset = i
        mask = (offset + tl.arange(0, BLOCK_N)) < N
        ptr = x_ptr + offset * stride_x
        x = tl.load(ptr, mask=mask, other=0.0)
        x = x - max_val
        exp_x = tl.exp(x)
        tl.store(y_ptr + offset * stride_y, exp_x, mask=mask)
        exp_sum = exp_sum + tl.sum(exp_x, axis=0)
        i += BLOCK_N
    tl.store(sum_ptr + row, exp_sum)

# Kernel to divide each exp(x - max) by the corresponding row sum
@triton.jit
def _softmax_row_divide(
    Y_ptr, sum_ptr, N, stride_y,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    y_ptr = Y_ptr + row * stride_y
    sum_val = tl.load(sum_ptr + row)
    i = 0
    while i < N:
        offset = i
        mask = (offset + tl.arange(0, BLOCK_N)) < N
        ptr = y_ptr + offset * stride_y
        exp_val = tl.load(ptr, mask=mask, other=0.0)
        softmax_val = exp_val / sum_val
        tl.store(ptr, softmax_val, mask=mask)
        i += BLOCK_N

def optimized_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Compute softmax along the last dimension of a 2‑D tensor using Triton.

    Args:
        x: Input tensor of