#!/usr/bin/env python3
"""
Optimised FFT Convolution solver.

This implementation replaces the reference scipy-based solution with a hand‑crafted
FFT routine that:
* Uses NumPy's real FFT (rfft/irfft) for real‑valued signals, halving the work.
* Pads to the next power‑of‑two for optimal FFT speed.
* Handles the three convolution modes ('full', 'same', 'valid') exactly as
  specified in the task description.
* Falls back to direct convolution for very short signals where FFT overhead
  would dominate.
* Returns the result as a Python list under the key 'convolution' so that
  the `is_solution` checker can validate it.

The solver is self‑contained and only depends on NumPy, which is available in
the execution environment.
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _next_pow_two(n: int) -> int:
    """Return the next power of two greater than or equal to n."""
    return 1 << (n - 1).bit_length()

def _fft_convolve(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the linear convolution of two real‑valued 1‑D arrays using FFT.
    The result is truncated to the true linear convolution length.
    """
    len_x, len_y = x.shape[0], y.shape[0]
    conv_len = len_x + len_y - 1

    # Small signals: direct convolution is cheaper
    if conv_len <= 64:
        return np.convolve(x, y, mode='full')

    # Pad to next power of two for efficient FFT
    n_fft = _next_pow_two(conv_len)

    # Real FFTs
    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)
    # Pointwise multiplication
    Z = X * Y
    # Inverse real FFT
    z_full = np.fft.irfft(Z, n=n_fft)

    # Truncate to the true linear convolution length
    return z_full[:conv_len]

def _slice_mode(z_full: np.ndarray, mode: str, len_x: int, len_y: int) -> np.ndarray:
    """
    Slice the full convolution result `z_full` according to the requested mode.
    """
    if mode == 'full':
        return z_full

    conv_len = len(z_full)

    if mode == 'same':
        out_len = max(len_x, len_y)
        start = (conv_len - out_len) // 2
        return z_full[start:start + out_len]

    if mode == 'valid':
        # Valid length is abs(len_x - len_y) + 1, start at min(len_x, len_y)-1
        out_len = abs(len_x - len_y) + 1
        if out_len <= 0:
            return np.array([], dtype=float)
        start = min(len_x, len_y) - 1
        return z_full[start:start + out_len]

    raise ValueError(f"Unsupported mode: {mode}")

# --------------------------------------------------------------------------- #
# Solver class
# --------------------------------------------------------------------------- #
class FFTConvolution:
    """
    Optimised FFT convolution solver.
    """

    def solve(self, problem: dict) -> dict:
        """
        Compute the convolution of two signals using FFT.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - 'signal_x': list of floats
                - 'signal_y': list of floats
                - 'mode': 'full', 'same' or 'valid' (default 'full')

        Returns
        -------
        dict
            Dictionary with key 'convolution' containing the result as a list.
        """
        # Extract inputs
        x = np.asarray(problem.get("signal_x", []), dtype=float)
        y = np.asarray(problem.get("signal_y", []), dtype=float)
        mode = problem.get("mode", "full")

        # Handle empty inputs early
        if x.size == 0 or y.size == 0:
            return {"convolution": []}

        # Compute full convolution via FFT
        z_full = _fft_convolve(x, y)

        # Slice according to mode
        z_out = _slice_mode(z_full, mode, x.size, y.size)

        return {"convolution": z_out.tolist()}

# --------------------------------------------------------------------------- #
# Entry point used by the evaluator
# --------------------------------------------------------------------------- #
def run_solver(problem: dict) -> dict:
    """
    Main entry point for the evaluator.

    Parameters
    ----------
    problem : dict
        Problem dictionary as described in the task.

    Returns
    -------
    dict
        Solution dictionary with key 'convolution'.
    """
    solver = FFTConvolution()
    return solver.solve(problem)