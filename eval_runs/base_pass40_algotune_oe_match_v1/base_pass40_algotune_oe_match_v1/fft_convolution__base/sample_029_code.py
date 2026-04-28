import logging
from typing import Dict, Any
import numpy as np
from scipy import signal

class FFTConvolution:
    """
    Optimised FFT convolution solver.
    """

    def __init__(self):
        pass

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the convolution of two real signals using the FFT approach.
        Supports 'full', 'same', and 'valid' modes.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
            - "signal_x": list of numbers
            - "signal_y": list of numbers
            - "mode": str, one of 'full', 'same', 'valid' (default 'full')

        Returns
        -------
        dict
            {"convolution": list of floats}
        """
        try:
            # Convert inputs to numpy arrays of float64
            x = np.asarray(problem.get("signal_x", []), dtype=np.float64)
            y = np.asarray(problem.get("signal_y", []), dtype=np.float64)
            mode = problem.get("mode", "full")

            len_x, len_y = x.size, y.size

            # Handle empty signals early
            if len_x == 0 or len_y == 0:
                return {"convolution": []}

            # ----- FFT-based full convolution --------------------------------
            # Zero‑pad to the full convolution length
            full_len = len_x + len_y - 1

            # Use real FFT (rfft/irfft) for speed when signals are real
            X = np.fft.rfft(x, n=full_len)
            Y = np.fft.rfft(y, n=full_len)
            conv_full = np.fft.irfft(X * Y, n=full_len)

            # ----- Select mode ------------------------------------------------
            if mode == "full":
                result = conv_full

            elif mode == "same":
                # Output length equals max(len_x, len_y)
                out_len = max(len_x, len_y)
                start = (full_len - out_len) // 2
                result = conv_full[start : start + out_len]

            elif mode == "valid":
                # Output length equals max(0, len_long - len_short + 1)
                out_len = max(0, max(len_x, len_y) - min(len_x, len_y) + 1)
                if out_len == 0:
                    result = np.empty(0, dtype=np.float64)
                else:
                    if len_x >= len_y:
                        start = len_y - 1
                    else:
                        start = len_x - 1
                    result = conv_full[start : start + out_len]

            else:
                raise ValueError(f"Unsupported convolution mode: {mode}")

            # Ensure the result is a Python list of floats
            return {"convolution": result.tolist()}

        except Exception as exc:
            logging.error(f"Error in FFTConvolution.solve: {exc}")
            raise

    def is_solution(self, problem: Dict[str, Any], solution: Dict[str, Any]) -> bool:
        """
        Validate the solution against scipy.signal.fftconvolve.
        """
        try:
            if "convolution" not in solution:
                logging.error("Solution missing 'convolution' key.")
                return False

            student_result = solution["convolution"]
            if not isinstance(student_result, list):
                logging.error("Convolution result must be a list.")
                return False

            student_arr = np.array(student_result, dtype=float)
            if not np.all(np.isfinite(student_arr)):
                logging.error("Convolution result contains non‑finite values.")
                return False

            x = np.asarray(problem["signal_x"], dtype=float)
            y = np.asarray(problem["signal_y"], dtype=float)
            mode = problem.get("mode", "full")

            # Expected length calculation
            len_x, len_y = len(x), len(y)
            if mode == "full":
                expected_len = len_x + len_y - 1
            elif mode == "same":
                expected_len = max(len_x, len_y)
            elif mode == "valid":
                expected_len = max(0, max(len_x, len_y) - min(len_x, len_y) + 1)
            else:
                logging.error(f"Invalid mode: {mode}")
                return False

            if len_x == 0 or len_y == 0:
                expected_len = 0

            if student_arr.size != expected_len:
                logging.error(
                    f"Incorrect length for mode '{mode}'. "
                    f"Expected {expected_len}, got {student_arr.size}."
                )
                return False

            # Reference solution
            reference = signal.fftconvolve(x, y, mode=mode)

            if not np.allclose(student_arr, reference, rtol=1e-6, atol=1e-6):
                diff = np.abs(student_arr - reference)
                logging.error(
                    f"Numerical difference exceeds tolerance. "
                    f"Max diff: {diff.max():.2e}, Avg diff: {diff.mean():.2e}"
                )
                return False

            return True

        except Exception as exc:
            logging.error(f"Error in is_solution: {exc}")
            return False


def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point used by the evaluator.
    """
    solver = FFTConvolution()
    return solver.solve(problem)