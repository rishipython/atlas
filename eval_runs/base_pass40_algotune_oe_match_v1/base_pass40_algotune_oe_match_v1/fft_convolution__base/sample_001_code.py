import numpy as np
from typing import Dict, Any

class FFTConvolution:
    """
    Optimized FFT convolution solver.
    Uses real FFT (rfft) for speed on real input signals.
    Handles 'full', 'same', and 'valid' modes with minimal padding.
    """

    def __init__(self):
        pass

    def _conv_full(self, x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
        """Compute full convolution via FFT."""
        # FFT of both signals with zero-padding to length n
        fx = np.fft.rfft(x, n)
        fy = np.fft.rfft(y, n)
        # Pointwise multiplication and inverse FFT
        return np.fft.irfft(fx * fy, n)

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Convert inputs to numpy arrays (float64 for consistency)
            x = np.asarray(problem.get("signal_x", []), dtype=np.float64)
            y = np.asarray(problem.get("signal_y", []), dtype=np.float64)
            mode = problem.get("mode", "full").lower()

            len_x, len_y = x.size, y.size

            # Handle empty inputs early
            if len_x == 0 or len_y == 0:
                return {"convolution": []}

            # Length of the full linear convolution
            n = len_x + len_y - 1

            # Compute full convolution once
            full_conv = self._conv_full(x, y, n)

            # Slice according to mode
            if mode == "full":
                result = full_conv
            elif mode == "same":
                # Return central part with length max(len_x, len_y)
                max_len = max(len_x, len_y)
                start = (n - max_len) // 2
                result = full_conv[start:start + max_len]
            elif mode == "valid":
                # Return only fully overlapping part
                min_len = min(len_x, len_y)
                start = min_len - 1
                valid_len = abs(len_x - len_y) + 1
                result = full_conv[start:start + valid_len]
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            # Convert to Python list for the output format
            return {"convolution": result.tolist()}

        except Exception as exc:
            # In a real system we might log the error; for this task we simply re-raise
            raise exc

def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point used by the evaluator.
    """
    solver = FFTConvolution()
    return solver.solve(problem)