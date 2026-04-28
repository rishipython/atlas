#!/usr/bin/env python3
"""
FFT Convolution Task – Optimised Implementation

This module provides a fast and numerically stable FFT‑based convolution
implementation for real‑valued signals.  It replaces the reference
implementation with a hand‑crafted routine that uses NumPy's real FFT
(`rfft`/`irfft`) and carefully slices the linear convolution to obtain
the requested mode (`full`, `same`, `valid`).  The implementation
avoids any external dependencies beyond NumPy and works for arbitrary
signal lengths, including empty inputs.

Author: OpenEvolve
"""

import numpy as np
from typing import Dict, Any


class FFTConvolution:
    """
    Optimised solver for the FFT convolution task.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the linear convolution of two real signals using the
        Fast Fourier Transform.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                - "signal_x": list or array of floats
                - "signal_y": list or array of floats
                - "mode": str, one of "full", "same", "valid"

        Returns
        -------
        dict
            Dictionary with key "convolution" containing the result
            as a list of floats.
        """
        # Extract signals and mode
        signal_x = np.asarray(problem.get("signal_x", []), dtype=np.float64)
        signal_y = np.asarray(problem.get("signal_y", []), dtype=np.float64)
        mode = problem.get("mode", "full")

        # Handle empty inputs immediately
        if signal_x.size == 0 or signal_y.size == 0:
            return {"convolution": []}

        nx, ny = signal_x.size, signal_y.size

        # Full convolution length
        n_full = nx + ny - 1

        # Next power of two for efficient FFT (optional but speeds up large arrays)
        fft_len = 1 << (n_full - 1).bit_length()

        # Compute FFTs using real transforms
        X = np.fft.rfft(signal_x, fft_len)
        Y = np.fft.rfft(signal_y, fft_len)

        # Element‑wise multiplication in frequency domain
        Z = X * Y

        # Inverse FFT to obtain linear convolution
        conv_full = np.fft.irfft(Z, fft_len)[:n_full]

        # Slice according to requested mode
        if mode == "full":
            result = conv_full
        elif mode == "same":
            # Centered segment of length nx
            start = (n_full - nx) // 2
            result = conv_full[start : start + nx]
        elif mode == "valid":
            # Valid part where signals fully overlap
            start = ny - 1
            valid_len = abs(nx - ny) + 1
            if valid_len <= 0:
                result = np.array([], dtype=np.float64)
            else:
                result = conv_full[start : start + valid_len]
        else:
            # Fallback to full if mode is unrecognised
            result = conv_full

        return {"convolution": result.tolist()}


def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point used by the evaluator.

    Parameters
    ----------
    problem : dict
        Problem definition dictionary.

    Returns
    -------
    dict
        Solver output dictionary.
    """
    solver = FFTConvolution()
    return solver.solve(problem)