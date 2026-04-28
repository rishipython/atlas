#!/usr/bin/env python3
"""
FFT Convolution Task

This implementation replaces the original scipy based solution with a
pure NumPy FFT based approach that is both faster and memory efficient
for large signals.  It handles the three standard convolution modes
('full', 'same', 'valid') and correctly deals with empty inputs.
"""

import logging
from typing import Dict, Any, List

import numpy as np


class FFTConvolution:
    """
    FFT based convolution solver.
    """

    def __init__(self):
        pass

    @staticmethod
    def _next_pow2(n: int) -> int:
        """Return the next power of two >= n."""
        return 1 << (n - 1).bit_length()

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Compute the convolution of two real-valued signals using FFT.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                - "signal_x": list of floats
                - "signal_y": list of floats
                - "mode": "full", "same" or "valid"

        Returns
        -------
        dict
            Dictionary with key "convolution" containing the result as a list.
        """
        try:
            # Extract data
            x = np.asarray(problem.get("signal_x", []), dtype=float)
            y = np.asarray(problem.get("signal_y", []), dtype=float)
            mode = problem.get("mode", "full")

            # Empty input handling
            if x.size == 0 or y.size == 0:
                return {"convolution": []}

            len_x, len_y = x.size, y.size

            # Full convolution length
            n_full = len_x + len_y - 1

            # Pad to next power of two for efficient FFT
            n_fft = self._next_pow2(n_full)

            # Use real FFT for real inputs
            X = np.fft.rfft(x, n_fft)
            Y = np.fft.rfft(y, n_fft)
            Z = X * Y
            full_conv = np.fft.irfft(Z, n_fft)[:n_full]  # keep only linear part

            if mode == "full":
                result = full_conv
            elif mode == "same":
                # Length of 'same' output is max(len_x, len_y)
                L = max(len_x, len_y)
                # Compute start index for slicing
                start = (min(len_x, len_y) - 1) // 2
                result = full_conv[start : start + L]
            elif mode == "valid":
                # Length of 'valid' output is |len_x - len_y| + 1
                if len_x >= len_y:
                    start, end = len_y - 1, len_x
                else:
                    start, end = len_x - 1, len_y
                result = full_conv[start:end]
            else:
                raise ValueError(f"Unsupported mode: {mode!r}")

            # Convert to Python list for JSON compatibility
            return {"convolution": result.tolist()}

        except Exception as exc:
            logging.error(f"Error in solve: {exc}")
            raise


def run_solver(problem: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Entry point for the evaluator.

    Parameters
    ----------
    problem : dict
        Problem definition dictionary.

    Returns
    -------
    dict
        Solution dictionary with key "convolution".
    """
    solver = FFTConvolution()
    return solver.solve(problem)