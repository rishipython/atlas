import numpy as np
import logging
from typing import Dict, Any

class FFTConvolution:
    """
    Optimized FFT Convolution solver.
    Uses rFFT for real signals and switches to direct convolution for very short inputs.
    """

    def __init__(self):
        pass

    @staticmethod
    def _next_pow_two(n: int) -> int:
        """Return the next power of two greater than or equal to n."""
        return 1 << (n - 1).bit_length()

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute convolution of two real signals using FFT.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - "signal_x": list or array of floats
                - "signal_y": list or array of floats
                - "mode": one of "full", "same", "valid"

        Returns
        -------
        dict
            Dictionary with key "convolution" containing the result as a list.
        """
        try:
            # Convert inputs to numpy arrays of float64
            x = np.asarray(problem.get("signal_x", []), dtype=np.float64)
            y = np.asarray(problem.get("signal_y", []), dtype=np.float64)
            mode = problem.get("mode", "full")

            # Handle empty signals
            if x.size == 0 or y.size == 0:
                return {"convolution": []}

            # For very short signals, direct convolution is cheaper
            if x.size + y.size <= 512:
                full = np.convolve(x, y)
            else:
                # Linear convolution via rFFT
                n_full = x.size + y.size - 1
                n_fft = self._next_pow_two(n_full)
                X = np.fft.rfft(x, n=n_fft)
                Y = np.fft.rfft(y, n=n_fft)
                Z = X * Y
                full = np.fft.irfft(Z, n=n_fft)[:n_full]

            # Slice according to mode
            if mode == "full":
                out = full
            elif mode == "same":
                out_len = max(x.size, y.size)
                start = (full.size - out_len) // 2
                out = full[start : start + out_len]
            elif mode == "valid":
                out_len = max(0, max(x.size, y.size) - min(x.size, y.size) + 1)
                if out_len == 0:
                    out = np.array([], dtype=np.float64)
                else:
                    start = (full.size - out_len) // 2
                    out = full[start : start + out_len]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            return {"convolution": out.tolist()}

        except Exception as exc:
            logging.error(f"Error in FFTConvolution.solve: {exc}")
            raise

    def is_solution(self, problem: Dict[str, Any], solution: Dict[str, Any]) -> bool:
        """
        Validation helper (kept for compatibility with the original framework).
        """
        try:
            if "convolution" not in solution:
                logging.error("Missing 'convolution' key in solution.")
                return False

            student = np.asarray(solution["convolution"], dtype=np.float64)
            if not np.all(np.isfinite(student)):
                logging.error("Convolution contains non-finite values.")
                return False

            x = np.asarray(problem.get("signal_x", []), dtype=np.float64)
            y = np.asarray(problem.get("signal_y", []), dtype=np.float64)
            mode = problem.get("mode", "full")

            # Expected length
            len_x, len_y = x.size, y.size
            if mode == "full":
                exp_len = len_x + len_y - 1
            elif mode == "same":
                exp_len = max(len_x, len_y)
            elif mode == "valid":
                exp_len = max(0, max(len_x, len_y) - min(len_x, len_y) + 1)
            else:
                logging.error(f"Invalid mode: {mode}")
                return False

            if exp_len != student.size:
                logging.error(f"Expected length {exp_len}, got {student.size}.")
                return False

            # Reference result
            from scipy import signal as _signal
            ref = _signal.fftconvolve(x, y, mode=mode)

            if not np.allclose(student, ref, rtol=1e-6, atol=1e-6):
                diff = np.abs(student - ref)
                logging.error(f"Numerical mismatch: max {diff.max():.2e}, mean {diff.mean():.2e}.")
                return False

            return True

        except Exception as exc:
            logging.error(f"Error in is_solution: {exc}")
            return False

def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for the solver.
    """
    solver = FFTConvolution()
    return solver.solve(problem)