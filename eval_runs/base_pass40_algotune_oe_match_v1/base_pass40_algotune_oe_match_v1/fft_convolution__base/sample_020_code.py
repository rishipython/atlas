import logging
import numpy as np
from scipy import signal
from typing import Dict, Any

# Attempt to import a fast length selector; fall back to power‑of‑two if unavailable
try:
    from scipy.fft import next_fast_len  # SciPy 1.4+ provides this
except Exception:
    def next_fast_len(n: int) -> int:
        """Return the next power of two greater than or equal to n."""
        return 1 << (n - 1).bit_length()


class FFTConvolution:
    """
    Optimised FFT convolution implementation.
    Supports 'full', 'same', and 'valid' modes.
    Uses direct convolution for small inputs to avoid FFT overhead.
    """

    SMALL_THRESHOLD = 5000  # Empirically chosen threshold for switching to direct conv

    def __init__(self):
        pass

    @staticmethod
    def _mode_indices(len_x: int, len_y: int, mode: str):
        """
        Compute the start index and output length for the requested mode.
        """
        conv_len = len_x + len_y - 1

        if mode == "full":
            return 0, conv_len

        if mode == "same":
            out_len = max(len_x, len_y)
            start = (conv_len - out_len) // 2
            return start, out_len

        if mode == "valid":
            if len_x >= len_y:
                start = len_y - 1
                out_len = len_x - len_y + 1
            else:
                start = len_x - 1
                out_len = len_y - len_x + 1
            return start, out_len

        raise ValueError(f"Unsupported mode: {mode}")

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the convolution of two real signals using an efficient FFT approach.
        """
        try:
            signal_x = np.asarray(problem["signal_x"], dtype=np.float64)
            signal_y = np.asarray(problem["signal_y"], dtype=np.float64)
            mode = problem.get("mode", "full")

            len_x, len_y = signal_x.size, signal_y.size

            # Handle empty inputs
            if len_x == 0 or len_y == 0:
                return {"convolution": []}

            # For very small signals use direct convolution
            if len_x * len_y <= self.SMALL_THRESHOLD:
                conv = np.convolve(signal_x, signal_y)
            else:
                conv_len = len_x + len_y - 1
                fft_len = next_fast_len(conv_len)

                # Perform real FFT convolution
                fx = np.fft.rfft(signal_x, n=fft_len)
                fy = np.fft.rfft(signal_y, n=fft_len)
                conv_full = np.fft.irfft(fx * fy, n=fft_len)

                # Trim to the true linear convolution length
                conv = conv_full[:conv_len]

            start, out_len = self._mode_indices(len_x, len_y, mode)
            result = conv[start:start + out_len]

            return {"convolution": result.tolist()}

        except Exception as e:
            logging.error(f"Error in solve method: {e}")
            raise


def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for the evaluator.
    """
    solver = FFTConvolution()
    return solver.solve(problem)