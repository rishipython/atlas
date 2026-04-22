import torch
import triton
import triton.language as tl

# Kernel that computes the per‑row mean and variance of a 2‑D tensor.
@triton.jit
def _mean_var_kernel(
    x: tl.tensor,
    mean: tl.tensor,
    var: tl.tensor,
    M: tl.int32,
    N: tl.int32,
    stride_x: tl.int32,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    row_sum = 0.0
    row_sq_sum = 0.0
    start = 0
    while start < N:
        block_start = start
        offset = row * stride_x + block_start
        mask = block_start + tl.arange(0, BLOCK) < N
        x_block = tl.load(x + offset, mask=mask, other=0.0)
        row_sum += tl.sum(x_block, axis=0)
        row_sq_sum += tl.sum(x_block * x_block, axis=0)
        start += BLOCK

    mean_val = row_sum / N
    var_val = row_sq_sum / N - mean_val * mean_val
    mean[row] = mean_val
    var[row] = var_val


# Kernel that normalises each row using the pre‑computed mean and variance.
@triton.jit
def _norm_kernel(
    x: tl.tensor,
    out: tl.tensor,
    weight: tl.tensor,
    bias: tl.tensor,
    mean: tl.tensor,
    var: tl.tensor,
    eps: tl.float32,
    M: tl.int32,
    N: tl.int32,
    stride_x: tl.int32,
    stride_out: tl.int32,
    stride_wb: tl.int32,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    mean_val = mean[row]
    var_val = var[row]
    std = tl.math.sqrt(var_val + eps)

    start = 0
    while start < N:
        block_start = start
        offset = row * stride_x + block_start
        mask = block_start + tl.arange(0, BLOCK) < N

        x_block = tl.load(x + offset, mask=mask, other=0.0)
        w_block = tl.load(weight + block_start, mask=mask, other=0.0)
        b_block = tl.load(bias + block_start, mask=mask, other=0.0)

        y_block = (x_block - mean_val) / std
        y_block = y_block * w_block + b_block

        out_ptr = row * stride_out + block_start
        tl.store(out + out_ptr, y_block, mask=mask)

        start += BLOCK


def optimized_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Optimised Layer Normalisation over the last dimension using Triton.

    Args:
        x (torch.Tensor): Input tensor of shape (M, N).
        weight (torch.Tensor): Scale parameter of shape (N,).
        bias (torch.Tensor): Shift parameter of shape (N,).
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.

    Returns:
        torch.Tensor: Normalised tensor of the same shape as `x`.
    """
    assert x.ndim == 2, "Input tensor must be 2-D"
    M, N = x.shape
    out = torch.empty_like(x)
    mean = torch.empty((M,), dtype=x.dtype, device=x.device)
    var = torch.empty((M,), dtype=x.dtype, device=x.device)

    BLOCK = 256  # Tune this for your GPU
    grid = (M,)

    # First pass: compute mean and variance
    _mean_var_kernel[grid](
        x,
        mean,
        var,
        M,
        N,
        x.stride(0),
        BLOCK=BLOCK,
    )

    # Second pass: normalise and apply affine transformation
    _norm_kernel[grid](
        x,
        out,
        weight,
        bias,
        mean,
        var,
        eps,
        M,
        N,
        x.stride(0),
        out.stride(0),
        weight.stride(0),
        BLOCK=BLOCK,
    )

    return out

# ------------------------------------------------------------
# Example usage and basic sanity check (uncomment to run)
# ------------------------------------------------------------
# if __name__ == "__main__":
#     torch.manual_seed(0)
#     for M, N in [(1024, 1024), (2048, 4096)]:
#         x = torch.randn(M, N, device="cuda", dtype=torch.float32)
#         weight = torch.randn(N, device="cuda", dtype=torch.float32)
#         bias = torch.randn(N, device="cuda", dtype=torch.float32)
#         out_ref = torch.nn.functional.layer_norm(x, [N], weight, bias, eps=1e-5)
#         out_opt = optimized_layernorm(x, weight, bias, eps=1e-5)
#         print(f"Shape {(M, N)}: close={torch.allclose(out_ref, out_opt, atol=0.01, rtol=0.01)}")