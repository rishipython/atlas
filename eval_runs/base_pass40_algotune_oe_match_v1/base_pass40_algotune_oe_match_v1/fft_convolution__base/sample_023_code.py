#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimised FFT Convolution Solver

This implementation replaces the reference solver with a highly efficient
version that uses NumPy's real FFT (rfft) and carefully handles the three
convolution modes: full, same, and valid. The algorithm:
    1. Computes the linear convolution via FFT with minimal zero‑padding
       (next power‑of‑two length).
    2. Extracts the required slice according to the requested mode.
    3. Returns the result as a plain Python list for compatibility with
       the evaluation harness.

The approach is fully vectorised, avoids Python loops, and works for
arbitrarily large real‑valued signals.
"""

import numpy as np
import logging

# --------------------------------------------------------------------------- #
# Solver implementation
# --------------------------------------------------------------------------- #
class FFTConvolution:
    """
    Optimised FFT Convolution solver.
    """

    def __init__(self):
        pass

    @staticmethod
    def _next_pow_two(n: int) -> int:
        """Return the next power of two greater than or equal to n."""
        return 1 << (n - 1).bit_length()

    def solve(self, problem):
        """
        Solve the convolution problem using a fast FFT approach.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - signal_x : list/array of numbers
                - signal_y : list/array of numbers
                - mode     : 'full', 'same', or 'valid' (default 'full')

        Returns
        -------
        dict
            {'convolution': list of floats}
        """
        try:
            # ---------------------------------------------------------------- #
            # Input extraction & validation
            # ---------------------------------------------------------------- #
            signal_x = np.asarray(problem.get("signal_x", []), dtype=float)
            signal_y = np.asarray(problem.get("signal_y", []), dtype=float)
            mode = problem.get("mode", "full").lower()

            len_x, len_y = signal_x.size, signal_y.size

            # Empty signals -> empty result
            if len_x == 0 or len_y == 0:
                return {"convolution": []}

            # ---------------------------------------------------------------- #
            # FFT based convolution
            # ---------------------------------------------------------------- #
            # Full linear convolution length
            full_len = len_x + len_y - 1
            # Choose optimal FFT size: next power of two
            n_fft = self._next_pow_two(full_len)

            # Real FFTs (real input signals)
            X = np.fft.rfft(signal_x, n_fft)
            Y = np.fft.rfft(signal_y, n_fft)

            # Element‑wise multiplication in frequency domain
            Z = X * Y

            # Inverse real FFT to obtain linear convolution
            conv_full = np.fft.irfft(Z, n_fft)

            # Keep only the linear part
            conv_full = conv_full[:full_len]

            # ---------------------------------------------------------------- #
            # Slice according to mode
            # ---------------------------------------------------------------- #
            if mode == "full":
                result = conv_full
            elif mode == "same":
                out_len = max(len_x, len_y)
                # Center the output: start index
                start = (full_len - out_len) // 2
                result = conv_full[start : start + out_len]
            elif mode == "valid":
                # Compute valid length
                if len_x >= len_y:
                    out_len = len_x - len_y + 1
                    start = len_y - 1
                else:
                    out_len = len_y - len_x + 1
                    start = len_x - 1
                if out_len <= 0:
                    result = np.array([], dtype=float)
                else:
                    result = conv_full[start : start + out_len]
            else:
                # Unknown mode -> default to full
                logging.warning(f"Unknown mode '{mode}', defaulting to 'full'.")
                result = conv_full

            return {"convolution": result.tolist()}

        except Exception as exc:
            logging.error(f"Error in FFTConvolution.solve: {exc}")
            raise

    # --------------------------------------------------------------------------- #
    # Validation helper (kept from reference for compatibility)
    # --------------------------------------------------------------------------- #
    def is_solution(self, problem, solution):
        """
        Validate the solution produced by solve.
        """
        # Re‑use the reference implementation for checking correctness
        from scipy import signal as sp_signal

        try:
            if "convolution" not in solution:
                logging.error("Missing 'convolution' key.")
                return False
            student = np.asarray(solution["convolution"], dtype=float)
            if not np.all(np.isfinite(student)):
                logging.error("Result contains non‑finite values.")
                return False

            signal_x = np.asarray(problem["signal_x"], dtype=float)
            signal_y = np.asarray(problem["signal_y"], dtype=float)
            mode = problem.get("mode", "full")

            ref = sp_signal.fftconvolve(signal_x, signal_y, mode=mode)

            if student.shape != ref.shape:
                logging.error(f"Length mismatch: expected {ref.shape}, got {student.shape}")
                return False

            if not np.allclose(student, ref, atol=1e-6, rtol=1e-6):
                logging.error("Numerical mismatch.")
                return False

            return True

        except Exception as exc:
            logging.error(f"is_solution error: {exc}")
            return False

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_solver(problem: dict) -> dict:
    """
    Entry point used by the evaluator.

    Parameters
    ----------
    problem : dict
        Problem definition for the FFT convolution task.

    Returns
    -------
    dict
        Solution dictionary containing the convolution result.
    """
    solver = FFTConvolution()
    return solver.solve(problem)

# --------------------------------------------------------------------------- #
# Example usage (uncomment to test)
# --------------------------------------------------------------------------- #
# if __name__ == "__main__":
#     example = {
#         "signal_x": [1.0, 2.0, 3.0, 4.0],
#         "signal_y": [5.0, 6.0, 7.0],
#         "mode": "full"
#     }
#     print(run_solver(example))