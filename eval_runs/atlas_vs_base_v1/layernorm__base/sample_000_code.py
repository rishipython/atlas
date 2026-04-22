```python
import torch
import triton
import triton.language as tl

@triton.jit
def _layernorm_kernel(
    x_ptr,
    weight_ptr