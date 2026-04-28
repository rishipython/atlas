import logging
import numpy as np
from scipy import signal
from typing import Dict

class FFTConvolution:
    """
    Optimised FFT Convolution implementation.
    Uses real FFT (rfft/irfft) and minimal zero‑padding.
    Handles 'full', 'same', and 'valid' modes.
    """

    def __init__(self):
        pass

    @staticmethod
    def _next_pow2(n: int) -> int:
        """Return the next power of two greater than or equal to n."""
        if n <= 1:
            return 1
        return 1 << (n - 1).bit_length()

    def solve(self, problem: Dict) -> Dict:
        """
        Compute convolution of two real signals using FFT.

        Parameters
        ----------
        problem : dict
            Dictionary with keys 'signal_x', 'signal_y', and optional 'mode'.

        Returns
        -------
        dict
            Dictionary with key 'convolution' containing the result as a list.
        """
        try:
            # Extract data
            x = np.asarray(problem.get("signal_x", []), dtype=float)
            y = np.asarray(problem.get("signal_y", []), dtype=float)
            mode = problem.get("mode", "full")

            # Handle empty input early
            if x.size == 0 or y.size == 0:
                return {"convolution": []}

            len_x, len_y = x.size, y.size

            # Full convolution length
            n_full = len_x + len_y - 1

            # FFT length (next power of two for speed)
            n_fft = self._next_pow2(n_full)

            # Compute FFTs
            X = np.fft.rfft(x, n=n_fft)
            Y = np.fft.rfft(y, n=n_fft)

            # Element‑wise multiplication
            Z = X * Y

            # Inverse FFT to obtain linear convolution
            conv_full = np.fft.irfft(Z, n=n_fft)[:n_full]

            # Slice according to mode
            if mode == "full":
                result = conv_full
            elif mode == "same":
                # Center part of length max(len_x, len_y)
                out_len = max(len_x, len_y)
                start = (n_full - out_len) // 2
                result = conv_full[start:start + out_len]
            elif mode == "valid":
                # Length when signals fully overlap
                out_len = max(0, min(len_x, len_y) - abs(len_x - len_y) + 1)
                if out_len == 0:
                    result = np.array([], dtype=float)
                else:
                    start = abs(len_x - len_y)
                    result = conv_full[start:start + out_len]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            return {"convolution": result.tolist()}

        except Exception as e:
            logging.error(f"Error in solve method: {e}")
            raise

    def is_solution(self, problem, solution):
        """
        Validate the solution against scipy's reference implementation.
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
                logging.error("Could not convert convolution result to a numeric array.")
                return False

            signal_x = np.array(problem["signal_x"])
            signal_y = np.array(problem["signal_y"])
            mode = problem.get("mode", "full")

            # Expected length calculation
            len_x, len_y = len(signal_x), len(signal_y)
            if mode == "full":
                expected_len = len_x + len_y - 1
            elif mode == "same":
                expected_len = max(len_x, len_y)
            elif mode == "valid":
                expected_len = max(0, min(len_x, len_y) - abs(len_x - len_y) + 1)
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

            # Reference computation
            try:
                reference = signal.fftconvolve(signal_x, signal_y, mode=mode)
            except Exception as e:
                logging.error(f"Reference calculation failed: {e}")
                return False

            # Allow small numerical differences
            if not np.allclose(student_result_np, reference, rtol=1e-6, atol=1e-6):
                diff = np.abs(student_result_np - reference)
                max_diff = np.max(diff) if diff.size else 0
                avg_diff = np.mean(diff) if diff.size else 0
                logging.error(
                    f"Numerical difference exceeds tolerance. "
                    f"Max diff: {max_diff:.2e}, Avg diff: {avg_diff:.2e}."
                )
                return False

            return True

        except Exception as e:
            logging.error(f"Error in is_solution: {e}")
            return False

def run_solver(problem: Dict) -> Dict:
    """
    Entry point used by the evaluator.
    """
    solver = FFTConvolution()
    return solver.solve(problem)