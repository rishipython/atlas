import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm_kernel(
    x, weight, bias, out,
    M: tl.constexpr, N: tl.constexpr,
    stride_x, stride_w, stride_b, stride_out,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Each program handles one row
    row = tl.program_id(0)
    if row >= M:
        return

    # --- compute mean and variance over the row ---
    sum_val = tl.zeros([1], dtype=tl.float32)[0]
    sum_sq = tl.zeros([1], dtype=tl.float32)[0]
    for start in range(0, N, BLOCK_N):
        offset = start
        mask = (offset + tl.arange(0, BLOCK_N) < N)
        ptr = x + row * stride_x + offset
        x_chunk = tl.load(ptr, mask=mask, other=0.0, dtype=tl.float32)
        sum_val += tl.sum(x_chunk, axis=0)
        sum_sq += tl.sum(x_chunk * x_chunk, axis=0)

    N_f = tl.full([1], N, dtype=tl.float32)[0]
    mean = sum_val / N_f
    var = sum_sq / N_f - mean * mean
    denom = tl.math.rsqrt(var + eps)

    # --- compute the normalized output ---
    for start in range(0, N, BLOCK_N):
        offset = start
        mask = (offset + tl.arange(0, BLOCK_N) < N)
        x_ptr = x + row * stride_x + offset
        w_ptr = weight + offset
        b_ptr = bias + offset
        out_ptr = out + row * stride_out + offset

        x_chunk = tl.load(x_ptr, mask=mask, other=0.0, dtype=tl.float32)
        w_chunk = tl.load(w_ptr, mask=mask, other=0.0, dtype=tl.float32)
        b_chunk = tl.load(b_ptr, mask=mask, other=0.0, dtype=tl.float32)

        out_chunk = ((x_chunk - mean) * denom) * w_chunk + b_chunk
        out_chunk = tl.cast(out_chunk, x.dtype)
        tl.store(out_ptr, out_chunk, mask=mask)


def optimized_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    LayerNorm over the last dimension of a 2-D tensor using Triton.

    Args:
        x: input tensor of shape (M, N)
        weight: learnable weight of shape (N,)
        bias: learnable bias of shape (N,)
        eps: small epsilon for numerical stability

    Returns:
        Tensor of same shape as x
    """
    device = x.device
    M, N = x.shape
    out = torch.empty_like(x, device=device)

    BLOCK_N = 128  # Tunable block size

    grid = (M,)
    _layernorm_kernel[grid](
        x, weight, bias, out,
        M, N,
        x.stride(0), weight.stride(0), bias.stride(0), out.stride(0),
        eps,
        BLOCK_N=BLOCK_N
    )
    return out


def reference_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, [x.shape[-1]], weight, bias, eps)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda")

    test_shapes = [{'M': 1024, 'N': 1024}, {'M': 2048, 'N': 4096}]
    for shape in test_shapes:
        M, N = shape['M'], shape['N']
        x = torch.randn(M, N, device=device, dtype=torch.float32)
        weight = torch.randn(N, device=device, dtype=torch.float32)
        bias = torch.randn(N, device=device, dtype=torch.float32)

        ref = reference_layernorm(x, weight, bias)
        triton_out = optimized_layernorm(x, weight, bias)

        max_diff = torch.max(torch.abs(ref - triton_out)).item()
        print(f"Shape {M}x{N}: max diff = {max_diff:.6f}")
        assert torch.allclose(ref, triton_out, atol=1e-2, rtol=1e-2), "Results differ!"
    print("All tests passed.")