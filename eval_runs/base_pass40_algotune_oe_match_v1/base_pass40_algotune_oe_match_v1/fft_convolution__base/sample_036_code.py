import logging
import numpy as np
from scipy import signal
from typing import Any, Dict, List, Optional

class FFTConvolution:
    """
    Optimised FFT convolution solver.
    Uses NumPy's real FFT for speed while preserving the behaviour of scipy.signal.fftconvolve.
    """

    def __init__(self):
        """No special initialisation required."""
        pass

    @staticmethod
    def _rfft_conv(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
        """
        Perform linear convolution of two real signals using rfft/irfft.
        Parameters
        ----------
        x, y : np.ndarray
            Real input signals.
        n : int
            Length of the output full convolution (len(x)+len(y)-1).
        Returns
        -------
        conv : np.ndarray
            Full linear convolution result of length n.
        """
        # Forward real FFTs with zero‑padding to length n
        fx = np.fft.rfft(x, n=n)
        fy = np.fft.rfft(y, n=n)
        # Pointwise multiplication
        fprod = fx * fy
        # Inverse real FFT to obtain full convolution
        conv = np.fft.irfft(fprod, n=n)
        return conv

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Compute the convolution of two real signals using FFT.
        The result is returned under the key 'convolution'.
        """
        try:
            signal_x = np.array(problem["signal_x"], dtype=float)
            signal_y = np.array(problem["signal_y"], dtype=float)
            mode = problem.get("mode", "full")

            len_x = len(signal_x)
            len_y = len(signal_y)

            # Handle empty inputs early
            if len_x == 0 or len_y == 0:
                return {"convolution": []}

            # Length of full linear convolution
            full_len = len_x + len_y - 1

            # Compute full convolution via rfft
            full_conv = self._rfft_conv(signal_x, signal_y, n=full_len)

            # Slice according to mode
            if mode == "full":
                result = full_conv
            elif mode == "same":
                # SciPy's 'same' returns the central part of length len_x
                start = (full_len - len_x) // 2
                result = full_conv[start : start + len_x]
            elif mode == "valid":
                # Valid output length: abs(len_x - len_y) + 1
                start = min(len_x, len_y) - 1
                valid_len = abs(len_x - len_y) + 1
                result = full_conv[start : start + valid_len]
            else:
                raise ValueError(f"Unknown convolution mode: {mode}")

            # Convert to list for JSON serialisation
            return {"convolution": result.tolist()}

        except Exception as e:
            logging.error(f"Error in solve method: {e}")
            raise

    def is_solution(self, problem: Dict[str, Any], solution: Dict[str, Any]) -> bool:
        """
        Validation logic unchanged from the reference implementation.
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
                    logging.error("Convolution result contains non-finite values (NaN or inf).")
                    return False
            except ValueError:
                logging.error("Could not convert convolution result to a numeric numpy array.")
                return False

            signal_x = np.array(problem["signal_x"])
            signal_y = np.array(problem["signal_y"])
            mode = problem.get("mode", "full")

            len_x = len(signal_x)
            len_y = len(signal_y)
            if mode == "full":
                expected_len = len_x + len_y - 1
            elif mode == "same":
                expected_len = len_x
            elif mode == "valid":
                expected_len = max(0, max(len_x, len_y) - min(len_x, len_y) + 1)
            else:
                logging.error(f"Invalid mode provided in problem: {mode}")
                return False

            if len_x == 0 or len_y == 0:
                expected_len = 0

            if len(student_result_np) != expected_len:
                logging.error(
                    f"Incorrect result length for mode '{mode}'. "
                    f"Expected {expected_len}, got {len(student_result_np)}."
                )
                return False

            try:
                reference_result = signal.fftconvolve(signal_x, signal_y, mode=mode)
            except Exception as e:
                logging.error(f"Error calculating reference solution: {e}")
                return False

            if expected_len == 0:
                if len(student_result_np) == 0:
                    return True
                else:
                    logging.error("Expected empty result for empty input, but got non-empty result.")
                    return False

            abs_tol = 1e-6
            rel_tol = 1e-6
            is_close = np.allclose(student_result_np, reference_result, rtol=rel_tol, atol=abs_tol)
            if not is_close:
                diff = np.abs(student_result_np - reference_result)
                max_diff = np.max(diff) if len(diff) > 0 else 0
                avg_diff = np.mean(diff) if len(diff) > 0 else 0
                logging.error(
                    f"Numerical difference between student solution and reference exceeds tolerance. "
                    f"Max diff: {max_diff:.2e}, Avg diff: {avg_diff:.2e} (atol={abs_tol}, rtol={rel_tol})."
                )
                return False

            return True
        except Exception as e:
            logging.error(f"Error in is_solution method: {e}")
            return False

def run_solver(problem: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Entry point for the solver.
    """
    solver = FFTConvolution()
    return solver.solve(problem)