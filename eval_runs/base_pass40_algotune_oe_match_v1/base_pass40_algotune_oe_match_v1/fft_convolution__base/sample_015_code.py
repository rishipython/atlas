#!/usr/bin/env python3
"""
Optimised FFT convolution solver.
"""

import logging
from typing import Dict, Any

import numpy as np
from scipy import signal

# Configure minimal logging
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")


class FFTConvolution:
    """
    Optimised implementation of FFT based convolution.
    """

    @staticmethod
    def _fft_convolve(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Perform linear convolution of two real 1‑D signals using the real FFT.
        The result is computed with the minimal FFT size that guarantees a
        linear convolution (no circular wrap‑around).
        """
        if x.size == 0 or y.size == 0:
            return np.array([], dtype=float)

        # Length of linear convolution
        out_len = x.size + y.size - 1
        # Choose an efficient FFT length (power‑of‑2 or next fast length)
        n = np.fft.next_fast_len(out_len)

        # Real FFTs
        X = np.fft.rfft(x, n)
        Y = np.fft.rfft(y, n)

        # Point‑wise multiplication
        Z = X * Y

        # Inverse real FFT and truncate to the exact linear convolution length
        z = np.fft.irfft(Z, n)[:out_len]
        return z

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the convolution problem using an efficient FFT approach.
        """
        try:
            signal_x = np.asarray(problem["signal_x"], dtype=float)
            signal_y = np.asarray(problem["signal_y"], dtype=float)
            mode = problem.get("mode", "full").lower()

            # Compute full convolution first
            full_conv = self._fft_convolve(signal_x, signal_y)

            # Handle modes
            if mode == "full":
                result = full_conv
            elif mode == "same":
                # Output length = max(len_x, len_y)
                out_len = max(signal_x.size, signal_y.size)
                start = (full_conv.size - out_len) // 2
                result = full_conv[start : start + out_len]
            elif mode == "valid":
                # Parts where signals fully overlap
                if signal_x.size >= signal_y.size:
                    out_len = signal_x.size - signal_y.size + 1
                    start = signal_y.size - 1
                else:
                    out_len = signal_y.size - signal_x.size + 1
                    start = signal_x.size - 1
                if out_len <= 0:
                    result = np.array([], dtype=float)
                else:
                    result = full_conv[start : start + out_len]
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            # Return as list to match the expected output format
            return {"convolution": result.tolist()}

        except Exception as exc:
            logging.error(f"Error in solve method: {exc}")
            raise


def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for the evaluator.
    """
    solver = FFTConvolution()
    return solver.solve(problem)