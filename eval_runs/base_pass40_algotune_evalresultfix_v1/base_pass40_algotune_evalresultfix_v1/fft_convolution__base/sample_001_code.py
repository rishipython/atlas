#!/usr/bin/env python3
"""
FFT Convolution Task

This module implements a fast FFT‑based convolution routine that supports the
three standard modes: 'full', 'same', and 'valid'.  It replaces the slower
scipy.signal.fftconvolve call with a hand‑crafted implementation that uses
NumPy's rFFT for real‑valued inputs, which is typically twice as fast.

Author: OpenAI ChatGPT
"""

import logging
import numpy as np

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _next_fast_len(n: int) -> int:
    """
    Return the next fast length for FFT of size >= n.
    Uses bit‑length to find the next power of two.
    """
    return 1 << (n - 1).bit_length()


def _fft_convolve(x: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    """
    Compute linear convolution of two real 1‑D signals using the FFT.

    Parameters
    ----------
    x, y : np.ndarray
        Real input signals.
    mode : str
        'full', 'same', or 'valid'.

    Returns
    -------
    np.ndarray
        Convolution result in the requested mode.
    """
    len_x, len_y = x.shape[0], y.shape[0]

    # Handle empty inputs
    if len_x == 0 or len_y == 0:
        return np.array([], dtype=float)

    # Length of the full linear convolution
    n_full = len_x + len_y - 1
    fft_len = _next_fast_len(n_full)

    # FFT of both signals (real FFT)
    X = np.fft.rfft(x, fft_len)
    Y = np.fft.rfft(y, fft_len)

    # Element‑wise multiplication in frequency domain
    Z = X * Y

    # Inverse real FFT to obtain the full convolution
    conv_full = np.fft.irfft(Z, fft_len)[:n_full]

    # Slice according to mode
    if mode == "full":
        return conv_full

    elif mode == "same":
        out_len = max(len_x, len_y)
        start = (n_full - out_len) // 2
        return conv_full[start : start + out_len]

    elif mode == "valid":
        out_len = abs(len_x - len_y) + 1
        start = min(len_x, len_y) - 1
        return conv_full[start : start + out_len]

    else:
        raise ValueError(f"Unsupported mode: {mode}")


# --------------------------------------------------------------------------- #
# Main Solver Class
# --------------------------------------------------------------------------- #

class FFTConvolution:
    """
    FFT based convolution solver.
    """

    def __init__(self):
        pass

    def solve(self, problem: dict) -> dict:
        """
        Compute the convolution of two signals using FFT.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - 'signal_x': list or array of numbers
                - 'signal_y': list or array of numbers
                - 'mode': 'full', 'same', or 'valid' (default 'full')

        Returns
        -------
        dict
            Dictionary with key 'convolution' containing the result as a list.
        """
        try:
            # Convert inputs to NumPy arrays (float64)
            signal_x = np.asarray(problem["signal_x"], dtype=float)
            signal_y = np.asarray(problem["signal_y"], dtype=float)
            mode = problem.get("mode", "full")

            # Validate mode
            if mode not in {"full", "same", "valid"}:
                raise ValueError(f"Invalid mode: {mode}")

            # Compute convolution
            conv = _fft_convolve(signal_x, signal_y, mode)

            # Return as list
            return {"convolution": conv.tolist()}

        except Exception as exc:
            logging.error(f"Error in solve: {exc}")
            raise

    # ----------------------------------------------------------------------- #
    # Validation helper (unchanged)
    # ----------------------------------------------------------------------- #

    def is_solution(self, problem, solution):
        """
        Validate the provided solution against the reference implementation.
        """
        try:
            if "convolution" not in solution:
                logging.error("Solution missing 'convolution' key.")
                return False

            student_result = solution["convolution"]

            if not isinstance(student_result, list):
                logging.error("Convolution result must be a list.")
                return False

            try:
                student_result_np = np.array(student_result, dtype=float)
                if not np.all(np.isfinite(student_result_np)):
                    logging.error("Convolution result contains non-finite values.")
                    return False
            except ValueError:
                logging.error("Could not convert convolution result to numeric array.")
                return False

            signal_x = np.array(problem["signal_x"], dtype=float)
            signal_y = np.array(problem["signal_y"], dtype=float)
            mode = problem.get("mode", "full")

            len_x, len_y = len(signal_x), len(signal_y)
            if mode == "full":
                expected_len = len_x + len_y - 1
            elif mode == "same":
                expected_len = max(len_x, len_y)
            elif mode == "valid":
                expected_len = max(0, abs(len_x - len_y) + 1)
            else:
                logging.error(f"Invalid mode: {mode}")
                return False

            if len_x == 0 or len_y == 0:
                expected_len = 0

            if len(student_result_np) != expected_len:
                logging.error(
                    f"Incorrect length: expected {expected_len}, got {len(student_result_np)}."
                )
                return False

            # Reference solution using scipy for validation
            from scipy import signal as sp_signal
            try:
                reference = sp_signal.fftconvolve(signal_x, signal_y, mode=mode)
            except Exception as e:
                logging.error(f"Reference calculation error: {e}")
                return False

            if not np.allclose(
                student_result_np, reference, rtol=1e-6, atol=1e-6
            ):
                diff = np.abs(student_result_np - reference)
                logging.error(
                    f"Numerical mismatch: max diff={diff.max():.2e}, mean diff={diff.mean():.2e}"
                )
                return False

            return True

        except Exception as exc:
            logging.error(f"Error in is_solution: {exc}")
            return False


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run_solver(problem: dict) -> dict:
    """
    Entry point used by the evaluation harness.
    """
    solver = FFTConvolution()
    return solver.solve(problem)

# --------------------------------------------------------------------------- #
# Example usage (uncomment to test)
# --------------------------------------------------------------------------- #
# if __name__ == "__main__":
#     prob = {
#         "signal_x": [1.0, 2.0, 3.0, 4.0],
#         "signal_y": [5.0, 6.0, 7.0],
#         "mode": "full",
#     }
#     print(run_solver(prob))