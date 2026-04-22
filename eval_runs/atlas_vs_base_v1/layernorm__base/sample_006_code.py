import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm_kernel(
    x_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    y_ptr: tl.tensor,
    M: tl.int32,
    N: tl.int32,
    stride_x: tl.int64,
    stride_y: tl.int64,
    stride_w: tl.int64,
    stride_b: tl.int64,
    eps: tl.float32,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    # First pass: compute sum and sum of squares
    sum_ = 0.0
    sumsq = 0.0
    for start in range(0, N, BLOCK_N):
        col_idx = tl.arange(0, BLOCK_N) + start
        mask = col_idx < N

        ptr = x_ptr + row * stride_x + col_idx
        x_chunk = tl.load(ptr, mask=mask, other=0.0, dtype=tl.float32)

        sum_ += tl.sum(x_chunk, axis=0)
        sumsq += tl.sum(x_chunk * x_chunk, axis=0)

    mean = sum_ / N
    var = sumsq / N - mean * mean
    inv_std = 1.0 / tl.math.sqrt(var + eps)

    # Second pass: compute the normalized output
    for start in range(0, N, BLOCK_N):
        col_idx = tl.arange(0, BLOCK_N) + start
        mask = col_idx < N

        ptr_x = x_ptr + row * stride_x + col_idx
        x_chunk = tl.load(ptr_x, mask=mask, other=0.0, dtype=tl.float32)

        ptr_w = weight_ptr + col_idx
        w_chunk = tl.load(ptr_w, mask=mask, other=0.0, dtype=tl.float32)

        ptr_b = bias_ptr + col_idx
        b_chunk = tl.load(ptr_b, mask=mask, other=0.0, dtype=tl.float32)

        y_chunk = (x_chunk - mean) * inv_std * w_chunk + b_chunk

        ptr_y = y_ptr + row * stride_y + col_idx
        tl.store(ptr_y, y_chunk, mask=mask)


def optimized_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Compute Layer Normalization over the last dimension of a 2‑D tensor using Triton.
    """
    assert x.ndim == 2
    assert weight.ndim == 1 and bias.ndim == 1
    assert weight.shape[0] == bias.shape[0] == x.shape[1]

    M, N = x.shape
    y = torch.empty_like(x)

    BLOCK_N = 128  # Tune for performance

    stride_x = x.stride(0)
    stride_y = y.stride(0)
    stride_w = weight.stride(0)
    stride_b = bias.stride(0)

    grid = (M,)

    _layernorm_kernel[grid](
        x,
        weight,
        bias,
        y,
        M,
        N,
        stride_x,
        stride_y,
        stride_w,
        stride_b,
        eps,
        BLOCK_N=BLOCK_N,
    )

    return y


# Reference implementation for testing
def reference_layernorm(x, weight, bias, eps=1e-5):
    return torch.nn.functional.layer_norm(x, [x.shape[-1]], weight, bias, eps)


if __name__ == "__main__":
    shapes = [{"M": 1024, "N": 1024}, {"M": 2048, "N": 4096}]
    for shape in shapes:
        M, N = shape["M"], shape["N"]
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        weight = torch.randn(N, device="cuda", dtype=torch.float32)
        bias = torch.randn(N, device="cuda", dtype=torch.float32)

        out_triton = optimized_layernorm(x, weight, bias)
        out_ref = reference_layernorm(x, weight, bias)

        print(f"Shape M={M}, N={N}")
        print("Max diff:", torch.max(torch.abs(out_triton - out_ref)).item())
        print("Mean diff:", torch.mean(torch.abs(out_triton - out_ref)).item())
        assert torch.allclose(out_triton, out_ref, atol=1e-2, rtol=1e-2)
    print("All tests passed.")