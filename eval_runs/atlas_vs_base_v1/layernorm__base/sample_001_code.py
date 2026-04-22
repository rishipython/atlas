```python
import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm_kernel(
    x,          # input tensor (M, N)
    weight,     # weight vector (N,)
    bias,       # bias vector (N,)
    y,          # output tensor (M, N)
    M,          # number of