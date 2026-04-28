import numpy as np
import logging
from scipy import signal
from typing import Dict, Any

# Optional: use numpy's fast length function if available
try:
    next_fast_len = np.fft.next_fast_len
except AttributeError:  # older numpy
    def next_fast_len(n):
        """Fallback to next power of two."""
        return 1 << (n - 1).bit_length()


class FFTConvolution:
    """
    Optimised FFT based convolution solver.
    """

    def __init__(self):
        pass

    @staticmethod
    def _fft_convolve(x: np.ndarray, y: np.ndarray, mode: str):
        """Perform linear convolution via real FFT."""
        len_x, len_y = len(x), len(y)
        if len_x == 0 or len_y == 0:
            return np.array([], dtype=float)

        # For very small sizes, direct convolution is faster
        if len_x * len_y <= 5000:
            return np.convolve(x, y, mode=mode)

        # Full linear convolution length
        n_full = len_x + len_y - 1
        fft_size = next_fast_len(n_full)

        # Perform real FFTs
        X = np.fft.rfft(x, n=fft_size)
        Y = np.fft.rfft(y, n=fft_size)

        # Element‑wise multiplication and inverse FFT
        conv_full = np.fft.irfft(X * Y, n=fft_size)[:n_full]

        # Slice according to mode
        if mode == "full":
            return conv_full
        elif mode == "same":
            # Centered part of length len_x
            start = (n_full - len_x) // 2
            return conv_full[start : start + len_x]
        elif mode == "valid":
            valid_len = abs(len_x - len_y) + 1
            if valid_len <= 0:
                return np.array([], dtype=float)
            start = max(len_x, len_y) - 1
            return conv_full[start : start + valid_len]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        try:
            signal_x = np.asarray(problem.get("signal_x", []), dtype=float)
            signal_y = np.asarray(problem.get("signal_y", []), dtype=float)
            mode = problem.get("mode", "full")

            conv = self._fft_convolve(signal_x, signal_y, mode)
            return {"convolution": conv.tolist()}
        except Exception as e:
            logging.error(f"Error in solve method: {e}")
            raise

    def is_solution(self, problem: Dict[str, Any], solution: Dict[str, Any]) -> bool:
        """
        Validation helper (unchanged from reference implementation).
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

            reference_result = signal.fftconvolve(signal_x, signal_y, mode=mode)

            if expected_len == 0:
                if len(student_result_np) == 0:
                    return True
                else:
                    logging.error("Expected empty result for empty input, but got non-empty result.")
                    return False

            is_close = np.allclose(student_result_np, reference_result, rtol=1e-6, atol=1e-6)
            if not is_close:
                diff = np.abs(student_result_np - reference_result)
                max_diff = np.max(diff) if len(diff) > 0 else 0
                avg_diff = np.mean(diff) if len(diff) > 0 else 0
                logging.error(
                    f"Numerical difference between student solution and reference exceeds tolerance. "
                    f"Max diff: {max_diff:.2e}, Avg diff: {avg_diff:.2e} (atol=1e-6, rtol=1e-6)."
                )
                return False

            return True
        except Exception as e:
            logging.error(f"Error in is_solution method: {e}")
            return False


def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for the evaluator.
    """
    solver = FFTConvolution()
    return solver.solve(problem)