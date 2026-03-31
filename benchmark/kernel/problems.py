"""Kernel optimization problems.

Each problem defines a reference PyTorch implementation that the agent
must rewrite as an optimized Triton kernel.  The evaluator runs
generated code in a subprocess and measures correctness + speedup.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KernelProblem:
    problem_id: str
    description: str
    reference_code: str        # full Python source defining the reference function
    entry_point: str           # function name the agent must define
    ref_entry_point: str       # function name in reference_code
    input_generator_code: str  # defines ``generate_inputs(**shape) -> list[Tensor]``
    atol: float = 1e-2
    rtol: float = 1e-2
    num_warmup: int = 10
    num_benchmark: int = 100
    test_shapes: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Problem 1 – Softmax
# ---------------------------------------------------------------------------
SOFTMAX = KernelProblem(
    problem_id="softmax",
    description=(
        "Write an optimized Triton kernel that computes softmax along the "
        "last dimension of a 2-D input tensor (shape M×N).  The kernel "
        "should handle arbitrary N (not just powers of two)."
    ),
    reference_code="""\
import torch

def reference_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=-1)
""",
    ref_entry_point="reference_softmax",
    entry_point="optimized_softmax",
    input_generator_code="""\
import torch

def generate_inputs(M=1024, N=1024, device="cuda", dtype=torch.float32, **kw):
    x = torch.randn(M, N, device=device, dtype=dtype)
    return [x]
""",
    test_shapes=[
        {"M": 1024, "N": 1024},
        {"M": 2048, "N": 4096},
        {"M": 256, "N": 8192},
    ],
)


# ---------------------------------------------------------------------------
# Problem 2 – Matrix Multiplication
# ---------------------------------------------------------------------------
MATMUL = KernelProblem(
    problem_id="matmul",
    description=(
        "Write an optimized Triton kernel that computes the matrix product "
        "C = A @ B for two 2-D float32 tensors.  A has shape (M, K) and B "
        "has shape (K, N)."
    ),
    reference_code="""\
import torch

def reference_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)
""",
    ref_entry_point="reference_matmul",
    entry_point="optimized_matmul",
    input_generator_code="""\
import torch

def generate_inputs(M=512, K=512, N=512, device="cuda", dtype=torch.float32, **kw):
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    return [A, B]
""",
    test_shapes=[
        {"M": 512, "K": 512, "N": 512},
        {"M": 1024, "K": 1024, "N": 1024},
    ],
    atol=1e-1,
    rtol=1e-1,
)

# ---------------------------------------------------------------------------
# Problem 3 – Layer Normalization
# ---------------------------------------------------------------------------
LAYERNORM = KernelProblem(
    problem_id="layernorm",
    description=(
        "Write an optimized Triton kernel that computes Layer Normalization "
        "over the last dimension of a 2-D input tensor (shape M×N).  "
        "The kernel receives the input tensor x, learnable weight and bias "
        "vectors (each of length N), and a small epsilon for numerical "
        "stability (default 1e-5)."
    ),
    reference_code="""\
import torch

def reference_layernorm(x, weight, bias, eps=1e-5):
    return torch.nn.functional.layer_norm(x, [x.shape[-1]], weight, bias, eps)
""",
    ref_entry_point="reference_layernorm",
    entry_point="optimized_layernorm",
    input_generator_code="""\
import torch

def generate_inputs(M=1024, N=1024, device="cuda", dtype=torch.float32, **kw):
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.ones(N, device=device, dtype=dtype)
    bias = torch.zeros(N, device=device, dtype=dtype)
    return [x, weight, bias]
""",
    test_shapes=[
        {"M": 1024, "N": 1024},
        {"M": 2048, "N": 4096},
    ],
)

KERNEL_PROBLEMS: dict[str, KernelProblem] = {
    p.problem_id: p for p in [SOFTMAX, MATMUL, LAYERNORM]
}
