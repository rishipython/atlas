import numpy as np
from scipy import signal
import logging

class FFTConvolution:
    """
    Fast FFT‑based convolution solver.
    """

    def __init__(self):
        pass

    @staticmethod
    def _next_pow2(n: int) -> int:
        """Return the next power of two greater than or equal to n."""
        return 1 << (n - 1).bit_length()

    @staticmethod
    def _convolve_fft(x: np.ndarray, y: np.ndarray, mode: str = "full") -> np.ndarray:
        """
        Perform linear convolution of real signals x and y using FFT.
        Supports 'full', 'same' and 'valid' modes.
        """
        len_x, len_y = x.size, y.size

        # Handle empty inputs
        if len_x == 0 or len_y == 0:
            return np.array([], dtype=float)

        # Full convolution length
        n_full = len_x + len_y - 1
        n_fft = FFTConvolution._next_pow2(n_full)

        # FFT of both signals (real FFT for speed)
        X = np.fft.rfft(x, n=n_fft)
        Y = np.fft.rfft(y, n=n_fft)
        Z = X * Y
        full = np.fft.irfft(Z, n=n_fft)[:n_full]

        if mode == "full":
            return full

        # Determine output length
        if mode == "same":
            out_len = max(len_x, len_y)
            start = (n_full - out_len) // 2
            return full[start:start + out_len]
        elif mode == "valid":
            if len_x >= len_y:
                valid_len = max(0, len_x - len_y + 1)
                start = len_y - 1
            else:
                valid_len = max(0, len_y - len_x + 1)
                start = len_x - 1
            if valid_len == 0:
                return np.array([], dtype=float)
            return full[start:start + valid_len]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def solve(self, problem: dict) -> dict:
        """
        Solve the convolution problem.
        """
        try:
            signal_x = np.array(problem["signal_x"], dtype=float)
            signal_y = np.array(problem["signal_y"], dtype=float)
            mode = problem.get("mode", "full")

            result = self._convolve_fft(signal_x, signal_y, mode=mode)

            return {"convolution": result.tolist()}
        except Exception as e:
            logging.error(f"Error in solve: {e}")
            raise

    def is_solution(self, problem: dict, solution: dict) -> bool:
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

            student_arr = np.array(student_result, dtype=float)
            if not np.all(np.isfinite(student_arr)):
                logging.error("Convolution result contains non-finite values.")
                return False

            signal_x = np.array(problem["signal_x"])
            signal_y = np.array(problem["signal_y"])
            mode = problem.get("mode", "full")

            # Expected length according to scipy
            len_x, len_y = len(signal_x), len(signal_y)
            if mode == "full":
                expected_len = len_x + len_y - 1
            elif mode == "same":
                expected_len = max(len_x, len_y)
            elif mode == "valid":
                if len_x >= len_y:
                    expected_len = max(0, len_x - len_y + 1)
                else:
                    expected_len = max(0, len_y - len_x + 1)
            else:
                logging.error(f"Invalid mode: {mode}")
                return False

            if len(student_arr) != expected_len:
                logging.error(
                    f"Incorrect result length for mode '{mode}'. "
                    f"Expected {expected_len}, got {len(student_arr)}."
                )
                return False

            # Reference result
            ref = signal.fftconvolve(signal_x, signal_y, mode=mode)
            if not np.allclose(student_arr, ref, rtol=1e-6, atol=1e-6):
                diff = np.abs(student_arr - ref)
                max_diff = np.max(diff) if diff.size else 0
                logging.error(
                    f"Result differs from reference. Max diff: {max_diff:.2e}"
                )
                return False

            return True
        except Exception as e:
            logging.error(f"Error in is_solution: {e}")
            return False


def run_solver(problem: dict) -> dict:
    """
    Entry point used by the evaluator.
    """
    solver = FFTConvolution()
    return solver.solve(problem)