import torch
import triton
import triton.language as tl

# Triton kernel that implements layer normalization over the last dimension
@triton.jit
def _layernorm_kernel(
    x_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    y_ptr: tl.tensor,
    M: tl.int32,
    N: tl.int32,
    stride_x: tl.int64,
    stride_w: tl.int64,
    stride_b: tl.int64,
    stride_y: tl.int64,
    eps: tl.float32,
    BLOCK_N: tl.constexpr,
    dtype: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    # --- 1st pass: compute mean and variance per row -----------------
    sum = 0.0
    sumsq = 0.0
    for start in range(0, N, BLOCK_N):
        offset = row * stride_x + start
        ptr = x_ptr + offset
        mask = start + tl.arange(0, BLOCK_N) < N
        x = tl.load(ptr, mask=mask, other=0.0)
        x = tl.cast(x, tl.float32)
        sum += tl.sum(x, axis=0)
        sumsq += tl.sum(x * x, axis=0)

    mean = sum / N
    var = sumsq / N - mean * mean
    denom = tl.rsqrt(var + eps)

    # --- 2nd pass: compute normalized output --------------------------
    for start in range(0, N, BLOCK_N):
        offset = row * stride_x + start
        ptr_x = x_ptr + offset
        ptr_y = y_ptr + offset
        mask = start + tl.arange(0, BLOCK_N) < N

        x = tl.load(ptr_x, mask=mask, other=0.0)
        x = tl.cast(x, tl.float32)

        w = tl.load(weight_ptr + start + tl.arange(0, BLOCK_N), mask=mask, other=0.0)
        w = tl.cast(w, tl.float32)

        b = tl.load(bias_ptr + start + tl.arange(0, BLOCK_N), mask=mask, other=0.0)
        b = tl.cast(b, tl.float32)

        norm = (x - mean) * denom
        y = norm * w + b
        y_out = tl.cast(y, dtype)

        tl.store(ptr_y, y_out, mask=mask)


def optimized_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Layer Normalization over the last dimension of a 2-D tensor using a Triton kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (M, N).
        weight (torch.Tensor): Learnable weight vector of shape (N,).
        bias (torch.Tensor): Learnable bias vector of shape (N,).
        eps (float, optional): Small value for numerical stability. Default: 1e-5.

    Returns:
        torch.Tensor: Normalized tensor with the same shape and dtype as `x`.
    """
    assert weight.shape == (x.shape[1],)
    assert bias.shape == (x.shape[1],)

    M, N = x.shape
    y = torch.empty_like(x)

    stride_x = x.stride(0)
    stride_w = weight.stride(0)
    stride_b = bias.stride(0)
    stride_y = y.stride(0)

    BLOCK_N = 128  # can be tuned for performance
    dtype_const = tl.float32 if x.dtype == torch.float32 else tl.float16

    grid = (M,)  # one program per row
    _layernorm_kernel[grid](
        x, weight, bias, y,
        M, N,
        stride_x, stride_w, stride_b, stride_y,
        eps,
        BLOCK_N=BLOCK_N,
        dtype=dtype_const
    )
    return y