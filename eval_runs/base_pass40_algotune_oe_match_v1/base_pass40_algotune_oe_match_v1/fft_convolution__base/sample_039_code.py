#!/usr/bin/env python3
"""
FFT Convolution Task – Optimised implementation

This file contains a fast, correct implementation of the FFT based convolution
task.  It replaces the original reference implementation by performing the
convolution with NumPy's real‑FFT routines, which are significantly faster
than the generic `scipy.signal.fftconvolve` for large real signals.

The public entry point for the evaluator is the `run_solver` function.
"""

import logging
import numpy as np
from typing import Dict, Any

# --------------------------------------------------------------------------- #
#  Helper: Fast FFT based convolution
# --------------------------------------------------------------------------- #
def _fft_convolve(x: np.ndarray, y: np.ndarray, mode: str = "full") -> np.ndarray:
    """
    Compute the linear convolution of two real signals using the real FFT.

    Parameters
    ----------
    x, y : np.ndarray
        Input 1‑D real signals.  They are converted to `float64` internally.
    mode : {"full", "same", "valid"}
        Convolution mode.  The behaviour matches that of
        `scipy.signal.fftconvolve` for the three supported modes.

    Returns
    -------
    np.ndarray
        Convolution result in the requested mode.  The array is of type
        `float64` and 1‑D.
    """
    # Handle empty inputs early
    if x.size == 0 or y.size == 0:
        return np.empty(0, dtype=np.float64)

    # Ensure we are working with float64
    x = np.asarray(x, dtype=np.float64, order="C")
    y = np.asarray(y, dtype=np.float64, order="C")

    len_x, len_y = x.size, y.size
    n_full = len_x + len_y - 1

    # Pad to the length needed for linear convolution
    # Use rfft/irfft for real‑valued signals (half the computation)
    X = np.fft.rfft(x, n=n_full)
    Y = np.fft.rfft(y, n=n_full)
    conv_full = np.fft.irfft(X * Y, n=n_full)

    # Slice according to mode
    if mode == "full":
        return conv_full
    elif mode == "same":
        # Center the result; length should match len_x (or len_y if longer)
        # This follows the convention of scipy.signal.fftconvolve
        start = (len_y - 1) // 2
        end = start + len_x
        return conv_full[start:end]
    elif mode == "valid":
        # Length = abs(len_x - len_y) + 1
        start = min(len_x, len_y) - 1
        length = abs(len_x - len_y) + 1
        return conv_full[start:start + length]
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Supported modes: 'full', 'same', 'valid'.")


# --------------------------------------------------------------------------- #
#  Solver class
# --------------------------------------------------------------------------- #
class FFTConvolution:
    """
    Optimised solver for the FFT convolution task.

    The solver implements the same interface as the reference implementation,
    but the `solve` method is replaced with a high‑performance FFT based
    computation.
    """

    def __init__(self):
        """No special initialisation required."""
        pass

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the convolution of two signals using FFT.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                "signal_x" : list[float]
                "signal_y" : list[float]
                "mode"     : str, one of "full", "same", "valid"

        Returns
        -------
        dict
            Dictionary with a single key `"convolution"` mapping to the
            convolution result as a list of floats.
        """
        try:
            # Extract inputs
            signal_x = problem["signal_x"]
            signal_y = problem["signal_y"]
            mode = problem.get("mode", "full")

            # Convert to NumPy arrays
            x_arr = np.asarray(signal_x, dtype=np.float64, order="C")
            y_arr = np.asarray(signal_y, dtype=np.float64, order="C")

            # Compute convolution
            conv = _fft_convolve(x_arr, y_arr, mode=mode)

            # Convert result to list for the expected output format
            return {"convolution": conv.tolist()}

        except Exception as exc:
            logging.error(f"Error in solve: {exc}")
            raise


# --------------------------------------------------------------------------- #
#  Entry point used by the evaluation harness
# --------------------------------------------------------------------------- #
def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function called by the evaluation harness.

    Parameters
    ----------
    problem : dict
        Problem definition as described in the task statement.

    Returns
    -------
    dict
        Solution dictionary containing the convolution result.
    """
    solver = FFTConvolution()
    return solver.solve(problem)