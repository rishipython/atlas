import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm_kernel(
    x_ptr: tl.tensor,
    w_ptr: tl.tensor,
    b_ptr: tl.tensor,
    out_ptr: tl.tensor,
    N: int,
    eps: float,
    stride_x: int,
    BLOCK: tl.constexpr,
):
    # Each block handles one row
    row = tl.program_id(0)

    # Base pointers for this row
    x_base = x_ptr + row * stride_x
    out_base = out_ptr + row * stride_x

    # First pass: compute mean and variance
    row_sum = 0.0
    row_sum_sq = 0.0
    for start in range(0, N, BLOCK):
        offset = start + tl.arange(0, BLOCK)
        mask = offset < N

        x_chunk = tl.load(x_base + offset * stride_x, mask=mask, other=0.0).to(tl.float32)
        row_sum += tl.sum(x_chunk, axis=0)
        row_sum_sq += tl.sum(x_chunk * x_chunk, axis=0)

    mean = row_sum / N
    var = row_sum_sq / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: compute normalized output
    for start in range(0, N, BLOCK):
        offset = start + tl.arange(0, BLOCK)
        mask = offset < N

        x_chunk = tl.load(x_base + offset * stride_x, mask=mask, other=0.0).to(tl.float32)
        norm = (x_chunk - mean) * inv_std

        w_chunk = tl.load(w_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        b_chunk = tl.load(b_ptr + offset, mask=mask, other=0.0).to(tl.float32)

        out_chunk = norm * w_chunk + b_chunk
        tl.store(out_base + offset * stride_x, out_chunk, mask=mask)


def optimized_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Compute layer normalization over the last dimension of a 2-D tensor using a Triton kernel.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (M, N).
    weight : torch.Tensor
        Weight vector of shape (N,).
    bias : torch.Tensor
        Bias vector of shape (N,).
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    torch.Tensor
        Normalized tensor of the same shape as `x`.
    """
    assert x.ndim == 2, "Input must be 2-D"
    assert weight.shape[0] == x.shape[1]
    assert bias.shape[0] == x.shape[1]

    M, N = x.shape
    out = torch.empty_like(x)

    BLOCK = 128
    grid = (M,)
    _layernorm_kernel[grid](
        x, weight, bias, out,
        N, eps, x.stride(0),
        BLOCK=BLOCK
    )
    return out


# Simple test harness
if __name__ == "__main__":
    torch.manual_seed(0)
    for cfg in [{'M': 1024, 'N': 1024}, {'M': 2048, 'N': 4096}]:
        M, N = cfg['M'], cfg['N']
        x = torch.randn(M, N, device="cuda")
        weight = torch.randn(N, device="cuda")
        bias = torch.randn(N, device="cuda")

        ref = torch.nn.functional.layer_norm(x, [N], weight, bias, eps=1e-5)
        out = optimized_layernorm(x, weight, bias, eps=1e-5)

        # Compare
        max_diff = (ref - out).abs().max().item()
        print(f"Shape {cfg} max diff: {max_diff:.6f}")
        assert torch.allclose(ref, out, atol=1e-2, rtol=1e-2), f"Mismatch for shape {cfg}"
    print("All tests passed.")