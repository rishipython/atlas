#!/usr/bin/env python3
"""
Optimised FFT Convolution solver.

The implementation uses NumPy's real‑FFT (`rfft`) to compute the linear
convolution efficiently.  It handles all three convolution modes
(`full`, `same`, `valid`) and takes care of empty inputs.

The solver returns a dictionary with the key ``"convolution"`` containing
the convolution result as a list of floats, matching the format expected
by the original reference implementation.
"""

import logging
import numpy as np
from numpy.fft import rfft, irfft, next_fast_len
from typing import Dict, Any

# --------------------------------------------------------------------------- #
# Solver implementation
# --------------------------------------------------------------------------- #
class FFTConvolution:
    """
    Optimised FFT convolution solver.
    """

    def __init__(self):
        """No state needed."""
        pass

    @staticmethod
    def _convolve_full(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the full linear convolution of two real signals using rFFT.

        Parameters
        ----------
        x, y : np.ndarray
            1‑D real input signals.

        Returns
        -------
        np.ndarray
            Full convolution result of length ``len(x)+len(y)-1``.
        """
        # Handle empty inputs early
        if x.size == 0 or y.size == 0:
            return np.array([], dtype=float)

        # Length of the linear convolution
        full_len = x.size + y.size - 1

        # Choose a fast FFT size (next power‑of‑two or nearest fast size)
        fft_size = next_fast_len(full_len)

        # Compute real FFTs
        X = rfft(x, n=fft_size)
        Y = rfft(y, n=fft_size)

        # Element‑wise multiplication in frequency domain
        Z = X * Y

        # Inverse real FFT to obtain convolution
        conv_full = irfft(Z, n=fft_size)

        # Truncate to the true linear convolution length
        return conv_full[:full_len]

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the FFT convolution problem.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                - "signal_x" : list or array of floats
                - "signal_y" : list or array of floats
                - "mode"     : "full", "same" or "valid" (default "full")

        Returns
        -------
        dict
            {"convolution": list of floats}
        """
        try:
            # Convert inputs to NumPy arrays
            signal_x = np.asarray(problem["signal_x"], dtype=float)
            signal_y = np.asarray(problem["signal_y"], dtype=float)
            mode = problem.get("mode", "full")

            # Validate mode
            if mode not in {"full", "same", "valid"}:
                raise ValueError(f"Unsupported mode: {mode}")

            # Compute full convolution once
            full_conv = self._convolve_full(signal_x, signal_y)

            # Determine output slice based on mode
            if mode == "full":
                result = full_conv
            elif mode == "same":
                # central part of length max(len(x), len(y))
                len_x, len_y = signal_x.size, signal_y.size
                out_len = max(len_x, len_y)
                start = (full_conv.size - out_len) // 2
                result = full_conv[start:start + out_len]
            else:  # mode == "valid"
                len_x, len_y = signal_x.size, signal_y.size
                if len_x >= len_y:
                    out_len = max(0, len_x - len_y + 1)
                else:
                    out_len = max(0, len_y - len_x + 1)
                if out_len == 0:
                    result = np.array([], dtype=float)
                else:
                    # valid part starts where both signals fully overlap
                    start = len_y - 1 if len_x >= len_y else len_x - 1
                    result = full_conv[start:start + out_len]

            # Convert to list for output
            return {"convolution": result.tolist()}

        except Exception as exc:
            logging.error(f"Error in solve: {exc}")
            raise

    # --------------------------------------------------------------------- #
    # Optional: Validation helper (kept for compatibility with original code)
    # --------------------------------------------------------------------- #
    @staticmethod
    def is_solution(problem: Dict[str, Any], solution: Dict[str, Any]) -> bool:
        """
        Validate a solution against the reference implementation.
        """
        try:
            if "convolution" not in solution:
                return False
            student = np.asarray(solution["convolution"], dtype=float)
            if not np.all(np.isfinite(student)):
                return False

            # Reference using scipy for correctness
            from scipy import signal as sp_signal
            ref = sp_signal.fftconvolve(
                np.asarray(problem["signal_x"], dtype=float),
                np.asarray(problem["signal_y"], dtype=float),
                mode=problem.get("mode", "full")
            )
            return np.allclose(student, ref, atol=1e-6, rtol=1e-6)
        except Exception:
            return False

# --------------------------------------------------------------------------- #
# Entry point used by the evaluator
# --------------------------------------------------------------------------- #
def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to run the solver.

    Parameters
    ----------
    problem : dict
        Problem definition.

    Returns
    -------
    dict
        Solution dictionary with key "convolution".
    """
    solver = FFTConvolution()
    return solver.solve(problem)

# --------------------------------------------------------------------------- #
# Example usage (uncomment to test manually)
# --------------------------------------------------------------------------- #
# if __name__ == "__main__":
#     prob = {
#         "signal_x": [1.0, 2.0, 3.0, 4.0],
#         "signal_y": [5.0, 6.0, 7.0],
#         "mode": "full"
#     }
#     print(run_solver(prob))