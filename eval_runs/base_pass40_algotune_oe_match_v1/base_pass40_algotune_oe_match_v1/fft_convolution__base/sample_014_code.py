#!/usr/bin/env python3
"""
FFT Convolution Task

This implementation focuses on speed and correctness for the convolution of two real signals
using the Fast Fourier Transform. It handles the three common modes ('full', 'same',
'valid') and automatically switches to direct convolution for very small signals where
FFT overhead dominates.

Author: OpenAI ChatGPT
"""

import logging
from typing import Dict, Any, List

import numpy as np
from scipy import signal


class FFTConvolution:
    """
    Optimized FFT convolution solver.
    """

    def __init__(self, direct_threshold: int = 64):
        """
        Parameters
        ----------
        direct_threshold : int
            If the maximum signal length is below this threshold, use direct convolution
            instead of FFT to avoid overhead.
        """
        self.direct_threshold = direct_threshold

    @staticmethod
    def _next_pow_two(n: int) -> int:
        """Return the next power of two greater than or equal to n."""
        return 1 << (n - 1).bit_length()

    def _fft_convolve(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Perform linear convolution via FFT using real FFT for efficiency.
        """
        len_out = len(x) + len(y) - 1
        nfft = self._next_pow_two(len_out)

        # Real FFT of zero-padded signals
        X = np.fft.rfft(x, n=nfft)
        Y = np.fft.rfft(y, n=nfft)

        # Element-wise multiplication in frequency domain
        Z = X * Y

        # Inverse real FFT to obtain time-domain convolution
        conv = np.fft.irfft(Z, n=nfft)

        # Truncate to exact output length
        return conv[:len_out]

    def _direct_convolve(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Direct convolution using NumPy's implementation."""
        return np.convolve(x, y)

    def _slice_mode(self, full_conv: np.ndarray, mode: str, len_x: int, len_y: int) -> np.ndarray:
        """
        Slice the full convolution result according to the requested mode.
        """
        len_out = len(full_conv)

        if mode == "full":
            return full_conv

        if mode == "same":
            out_len = max(len_x, len_y)
            start = (len_out - out_len) // 2
            return full_conv[start : start + out_len]

        if mode == "valid":
            out_len = max(0, len_x - len_y + 1)
            if out_len == 0:
                return np.array([], dtype=float)
            start = (len_out - out_len) // 2
            return full_conv[start : start + out_len]

        # Fallback: return full (should not reach here)
        return full_conv

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Compute the convolution of two signals using FFT or direct method.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - "signal_x": list of numbers
                - "signal_y": list of numbers
                - "mode": one of "full", "same", "valid" (default "full")

        Returns
        -------
        dict
            {"convolution": [float, ...]}
        """
        try:
            # Input extraction
            signal_x = np.asarray(problem.get("signal_x", []), dtype=np.float64)
            signal_y = np.asarray(problem.get("signal_y", []), dtype=np.float64)
            mode = problem.get("mode", "full")

            len_x, len_y = len(signal_x), len(signal_y)

            # Handle empty inputs
            if len_x == 0 or len_y == 0:
                return {"convolution": []}

            # Decide on method
            if max(len_x, len_y) < self.direct_threshold:
                conv_full = self._direct_convolve(signal_x, signal_y)
            else:
                conv_full = self._fft_convolve(signal_x, signal_y)

            # Slice according to mode
            conv_mode = self._slice_mode(conv_full, mode, len_x, len_y)

            # Convert to list of Python floats
            result_list = conv_mode.tolist()
            return {"convolution": result_list}

        except Exception as exc:
            logging.error(f"Error in solve method: {exc}")
            raise


def run_solver(problem: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Entry point for the evaluator.

    Parameters
    ----------
    problem : dict
        Problem definition as described in the task.

    Returns
    -------
    dict
        The solution dictionary containing the convolution result.
    """
    solver = FFTConvolution()
    return solver.solve(problem)