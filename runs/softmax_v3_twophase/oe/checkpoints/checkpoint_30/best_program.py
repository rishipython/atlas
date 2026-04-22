"""Starting program for OpenEvolve — softmax.

Task: Write an optimized Triton kernel that computes softmax along the last dimension of a 2-D input tensor (shape M×N).  The kernel should handle arbitrary N (not just powers of two).

Requirements for the evolved program:
  - Define a function named exactly `optimized_softmax` with the same
    signature as the reference below.
  - Numerical output must be close to the reference (atol=0.01, rtol=0.01).
  - Test shapes the evaluator will run: [{'M': 1024, 'N': 1024}, {'M': 2048, 'N': 4096}, {'M': 256, 'N': 8192}].

The current body is the reference PyTorch implementation (correct but slow).
Evolve it into a Triton kernel.  Inside the evolve block you may add any
helper functions, `@triton.jit` kernels, imports, constants, etc. — as long as
`optimized_softmax` remains defined with the same signature.
"""

# EVOLVE-BLOCK-START
import torch

def optimized_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=-1)
# EVOLVE-BLOCK-END
