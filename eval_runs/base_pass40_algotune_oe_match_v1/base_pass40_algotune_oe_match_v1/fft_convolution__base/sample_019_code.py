import numpy as np
import logging
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

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the convolution of two real signals using FFT.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                - "signal_x": list[float]
                - "signal_y": list[float]
                - "mode": str, one of "full", "same", "valid"

        Returns
        -------
        dict
            Dictionary with key "convolution" containing the result as a list.
        """
        try:
            x = np.asarray(problem["signal_x"], dtype=np.float64)
            y = np.asarray(problem["signal_y"], dtype=np.float64)
            mode = problem.get("mode", "full")

            len_x, len_y = x.size, y.size

            # Handle empty input early
            if len_x == 0 or len_y == 0:
                return {"convolution": []}

            # Full linear convolution length
            full_len = len_x + len_y - 1

            # Determine FFT size (next power of two for speed)
            nfft = self._next_pow2(full_len)

            # Forward FFTs (real)
            X = np.fft.rfft(x, n=nfft)
            Y = np.fft.rfft(y, n=nfft)

            # Element‑wise multiplication in frequency domain
            conv_full = np.fft.irfft(X * Y, n=nfft)[:full_len]

            # Slice according to mode
            if mode == "full":
                result = conv_full
            elif mode == "same":
                out_len = max(len_x, len_y)
                start = (full_len - out_len) // 2
                result = conv_full[start : start + out_len]
            elif mode == "valid":
                # Valid length: max(0, abs(len_x - len_y) + 1)
                out_len = max(0, abs(len_x - len_y) + 1)
                if out_len == 0:
                    result = np.array([], dtype=np.float64)
                else:
                    start = min(len_x, len_y) - 1
                    result = conv_full[start : start + out_len]
            else:
                raise ValueError(f"Unsupported mode: {mode!r}")

            return {"convolution": result.tolist()}

        except Exception as exc:
            logging.error(f"Error in solve: {exc}")
            raise

    def is_solution(self, problem: Dict[str, Any], solution: Dict[str, Any]) -> bool:
        """
        Validate the solution produced by solve().
        """
        try:
            if "convolution" not in solution:
                logging.error("Missing 'convolution' key.")
                return False

            student = np.asarray(solution["convolution"], dtype=np.float64)
            if not np.all(np.isfinite(student)):
                logging.error("Result contains non‑finite values.")
                return False

            x = np.asarray(problem["signal_x"], dtype=np.float64)
            y = np.asarray(problem["signal_y"], dtype=np.float64)
            mode = problem.get("mode", "full")

            len_x, len_y = x.size, y.size
            if len_x == 0 or len_y == 0:
                expected_len = 0
            else:
                if mode == "full":
                    expected_len = len_x + len_y - 1
                elif mode == "same":
                    expected_len = max(len_x, len_y)
                elif mode == "valid":
                    expected_len = max(0, abs(len_x - len_y) + 1)
                else:
                    logging.error(f"Unknown mode: {mode}")
                    return False

            if student.size != expected_len:
                logging.error(f"Expected length {expected_len}, got {student.size}")
                return False

            # Reference using scipy for verification
            from scipy import signal
            ref = signal.fftconvolve(x, y, mode=mode)
            if not np.allclose(student, ref, rtol=1e-6, atol=1e-6):
                logging.error("Result differs from reference.")
                return False

            return True

        except Exception as exc:
            logging.error(f"is_solution error: {exc}")
            return False

def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point used by the evaluator.
    """
    solver = FFTConvolution()
    return solver.solve(problem)