import logging
import numpy as np
from scipy import signal
from typing import Any, Dict, List

class FFTConvolution:
    """
    Optimized FFT convolution solver.
    """

    def __init__(self):
        pass

    @staticmethod
    def _next_pow2(n: int) -> int:
        """Return the next power of two greater than or equal to n."""
        return 1 << (n - 1).bit_length()

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Compute the convolution of two real-valued signals using FFT.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                - "signal_x": list or array of real numbers
                - "signal_y": list or array of real numbers
                - "mode": "full", "same", or "valid" (default: "full")

        Returns
        -------
        dict
            {"convolution": list_of_numbers}
        """
        try:
            # Extract and convert signals
            x = np.asarray(problem.get("signal_x", []), dtype=np.float64)
            y = np.asarray(problem.get("signal_y", []), dtype=np.float64)
            mode = problem.get("mode", "full").lower()

            # Handle empty inputs early
            if x.size == 0 or y.size == 0:
                return {"convolution": []}

            # Determine output length and slice indices
            len_x, len_y = x.size, y.size
            full_len = len_x + len_y - 1

            if mode == "full":
                out_len = full_len
                start = 0
            elif mode == "same":
                out_len = len_x
                start = (len_y - 1) // 2
            elif mode == "valid":
                out_len = max(0, len_x - len_y + 1)
                start = len_y - 1
            else:
                raise ValueError(f"Unsupported mode '{mode}'")

            if out_len == 0:
                return {"convolution": []}

            # FFT size: next power of two >= full_len
            n_fft = self._next_pow2(full_len)

            # Zero‑pad and compute FFTs using real transforms
            X = np.fft.rfft(x, n=n_fft)
            Y = np.fft.rfft(y, n=n_fft)

            # Point‑wise multiplication and inverse FFT
            Z = np.fft.irfft(X * Y, n=n_fft)

            # Slice the desired portion
            result = Z[start : start + out_len]

            # Convert to list for the required output format
            return {"convolution": result.tolist()}

        except Exception as e:
            logging.error(f"Error in solve method: {e}")
            raise

    def is_solution(self, problem: Dict[str, Any], solution: Dict[str, Any]) -> bool:
        """
        Validate the solution produced by solve().
        """
        try:
            if "convolution" not in solution:
                logging.error("Missing key 'convolution'")
                return False

            student = solution["convolution"]
            if not isinstance(student, list):
                logging.error("Result must be a list")
                return False

            # Convert to numpy array for numerical checks
            student_np = np.array(student, dtype=np.float64)
            if not np.all(np.isfinite(student_np)):
                logging.error("Non‑finite values in result")
                return False

            # Reference solution using scipy.signal.fftconvolve
            x = np.asarray(problem.get("signal_x", []), dtype=np.float64)
            y = np.asarray(problem.get("signal_y", []), dtype=np.float64)
            mode = problem.get("mode", "full")

            ref = signal.fftconvolve(x, y, mode=mode)
            if ref.size != student_np.size:
                logging.error(f"Length mismatch: expected {ref.size}, got {student_np.size}")
                return False

            if not np.allclose(student_np, ref, rtol=1e-6, atol=1e-6):
                diff = np.abs(student_np - ref)
                logging.error(f"Numerical mismatch: max diff {diff.max():.2e}")
                return False

            return True
        except Exception as e:
            logging.error(f"Error in is_solution: {e}")
            return False

def run_solver(problem: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Entry point for the evaluator.

    Parameters
    ----------
    problem : dict
        Problem dictionary as described in the task.

    Returns
    -------
    dict
        Solution dictionary with key 'convolution'.
    """
    solver = FFTConvolution()
    return solver.solve(problem)