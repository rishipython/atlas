import numpy as np
from scipy import signal
import logging
from typing import Dict, Any

# --------------------------------------------------------------------------- #
# Optimised FFT Convolution
# --------------------------------------------------------------------------- #
def _next_pow2(n: int) -> int:
    """Return the next power of two >= n."""
    return 1 << (n - 1).bit_length()

def _fast_fft_convolve(x: np.ndarray, y: np.ndarray, mode: str = "full") -> np.ndarray:
    """
    Perform linear convolution of real signals x and y using FFT with minimal padding.
    Handles the three standard modes: 'full', 'same', 'valid'.
    """
    # Edge cases: empty signals
    if x.size == 0 or y.size == 0:
        return np.array([], dtype=float)

    # Direct convolution for very small signals to avoid FFT overhead
    if x.size * y.size < 5000:
        return np.convolve(x, y, mode=mode)

    # Full convolution length
    L = x.size + y.size - 1
    n_fft = _next_pow2(L)

    # Real FFT (rfft) for efficiency
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    Z = X * Y
    conv_full = np.fft.irfft(Z, n=n_fft)[:L]  # trim to exact length

    # Return the requested mode
    if mode == "full":
        return conv_full
    elif mode == "same":
        # center part of length max(len(x), len(y))
        target_len = max(x.size, y.size)
        start = (L - target_len) // 2
        return conv_full[start:start + target_len]
    elif mode == "valid":
        # length max(len_x, len_y) - min(len_x, len_y) + 1
        target_len = max(x.size, y.size) - min(x.size, y.size) + 1
        if target_len <= 0:
            return np.array([], dtype=float)
        start = (L - target_len) // 2
        return conv_full[start:start + target_len]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

# --------------------------------------------------------------------------- #
# Solver Class
# --------------------------------------------------------------------------- #
class FFTConvolution:
    """
    Optimised FFT Convolution solver.
    """

    def __init__(self):
        pass

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the convolution of two real signals using FFT.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - 'signal_x': list[float]
                - 'signal_y': list[float]
                - 'mode'    : str, one of 'full', 'same', 'valid'

        Returns
        -------
        dict
            {'convolution': list[float]}
        """
        try:
            x = np.asarray(problem["signal_x"], dtype=float)
            y = np.asarray(problem["signal_y"], dtype=float)
            mode = problem.get("mode", "full")

            # Validate mode
            if mode not in {"full", "same", "valid"}:
                raise ValueError(f"Invalid mode: {mode}")

            conv = _fast_fft_convolve(x, y, mode=mode)
            return {"convolution": conv.tolist()}

        except Exception as exc:
            logging.error(f"Error in solve: {exc}")
            raise

    # The is_solution method is unchanged; it will be used by the evaluator.

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point used by the evaluation harness.

    Parameters
    ----------
    problem : dict
        Problem definition as described in the task.

    Returns
    -------
    dict
        Solution dictionary containing the convolution result.
    """
    solver = FFTConvolution()
    return solver.solve(problem)