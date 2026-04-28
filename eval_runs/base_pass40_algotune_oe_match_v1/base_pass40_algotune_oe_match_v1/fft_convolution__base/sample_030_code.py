import numpy as np
import logging
from scipy import signal
from typing import Dict, Any

class FFTConvolution:
    """
    Optimised FFT convolution solver.
    """

    def __init__(self):
        pass

    @staticmethod
    def _next_pow2(n: int) -> int:
        """Return the next power of two >= n."""
        return 1 << (n - 1).bit_length()

    @staticmethod
    def _fftconvolve(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
        """
        Perform linear convolution of two real 1‑D arrays using FFT.
        Uses rfft/irfft for speed.
        """
        # Handle empty inputs
        if a.size == 0 or b.size == 0:
            return np.array([], dtype=float)

        # Full convolution length
        n_full = a.size + b.size - 1
        n_fft = FFTConvolution._next_pow2(n_full)

        # FFT of both signals
        fa = np.fft.rfft(a, n=n_fft)
        fb = np.fft.rfft(b, n=n_fft)

        # Point‑wise multiplication
        fc = fa * fb

        # Inverse FFT to get linear convolution
        conv = np.fft.irfft(fc, n=n_fft)[:n_full]

        if mode == "full":
            return conv
        elif mode == "same":
            # Length of the output is the maximum of input lengths
            out_len = max(a.size, b.size)
            start = (n_full - out_len) // 2
            return conv[start : start + out_len]
        elif mode == "valid":
            # Length of valid part
            if a.size >= b.size:
                out_len = a.size - b.size + 1
            else:
                out_len = b.size - a.size + 1
            if out_len <= 0:
                return np.array([], dtype=float)
            start = max(a.size, b.size) - min(a.size, b.size)
            return conv[start : start + out_len]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute convolution of two signals using an efficient FFT implementation.
        """
        try:
            signal_x = np.asarray(problem["signal_x"], dtype=float)
            signal_y = np.asarray(problem["signal_y"], dtype=float)
            mode = problem.get("mode", "full")

            conv_result = self._fftconvolve(signal_x, signal_y, mode)

            return {"convolution": conv_result.tolist()}
        except Exception as e:
            logging.error(f"Error in solve method: {e}")
            raise

    def is_solution(self, problem: Dict[str, Any], solution: Dict[str, Any]) -> bool:
        """
        Validate the convolution result against scipy's reference implementation.
        """
        try:
            if "convolution" not in solution:
                logging.error("Solution missing 'convolution' key.")
                return False

            student_result = solution["convolution"]
            if not isinstance(student_result, list):
                logging.error("Convolution result must be a list.")
                return False

            student_result_np = np.array(student_result, dtype=float)
            if not np.all(np.isfinite(student_result_np)):
                logging.error("Convolution result contains non-finite values.")
                return False

            signal_x = np.array(problem["signal_x"])
            signal_y = np.array(problem["signal_y"])
            mode = problem.get("mode", "full")

            # Expected length
            len_x, len_y = len(signal_x), len(signal_y)
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

            if student_result_np.size != expected_len:
                logging.error(
                    f"Incorrect result length for mode '{mode}'. "
                    f"Expected {expected_len}, got {student_result_np.size}."
                )
                return False

            # Reference using scipy
            reference = signal.fftconvolve(signal_x, signal_y, mode=mode)

            # Numerical check
            if not np.allclose(student_result_np, reference, rtol=1e-6, atol=1e-6):
                diff = np.abs(student_result_np - reference)
                max_diff = diff.max() if diff.size else 0
                logging.error(
                    f"Result differs from reference. Max diff: {max_diff:.2e}"
                )
                return False

            return True
        except Exception as e:
            logging.error(f"Error in is_solution: {e}")
            return False


def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for the evaluator.
    """
    solver = FFTConvolution()
    return solver.solve(problem)