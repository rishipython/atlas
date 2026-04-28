import numpy as np
from scipy import signal

# --------------------------------------------------------------------------- #
#  FFT Convolution solver (optimized for speed and correctness)
# --------------------------------------------------------------------------- #

class FFTConvolution:
    """
    Optimised FFT based convolution implementation.
    Supports 'full', 'same' and 'valid' modes.
    Uses real FFT (rfft/irfft) for real-valued inputs and
    falls back to direct convolution for very short signals.
    """

    # Threshold below which direct convolution is faster than FFT
    _direct_threshold = 512

    def _convolve_fft(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Perform linear convolution of two real 1‑D arrays using rfft.
        """
        len_x, len_y = x.size, y.size
        # Zero‑pad to length n = len_x + len_y - 1
        n = len_x + len_y - 1
        if n == 0:
            return np.empty(0, dtype=float)

        # Choose FFT length as next power of two for speed
        n_fft = 1 << (n - 1).bit_length()

        # Compute FFTs
        X = np.fft.rfft(x, n=n_fft)
        Y = np.fft.rfft(y, n=n_fft)

        # Element‑wise multiplication and inverse FFT
        conv_full = np.fft.irfft(X * Y, n=n_fft)

        # Truncate to the true linear convolution length
        return conv_full[:n].astype(float)

    def _convolve_direct(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Direct convolution using NumPy (no FFT).
        """
        return np.convolve(x, y, mode='full')

    def solve(self, problem: dict) -> dict:
        """
        Solve the FFT convolution problem.

        Parameters
        ----------
        problem : dict
            Dictionary with keys 'signal_x', 'signal_y' and 'mode'.

        Returns
        -------
        dict
            Dictionary with key 'convolution' containing the result list.
        """
        # Extract inputs
        x = np.asarray(problem.get("signal_x", []), dtype=float)
        y = np.asarray(problem.get("signal_y", []), dtype=float)
        mode = problem.get("mode", "full").lower()

        # Handle empty inputs early
        if x.size == 0 or y.size == 0:
            return {"convolution": []}

        # Choose method
        if max(x.size, y.size) <= self._direct_threshold:
            conv_full = self._convolve_direct(x, y)
        else:
            conv_full = self._convolve_fft(x, y)

        # Slice according to mode
        if mode == "full":
            result = conv_full
        elif mode == "same":
            len_x = x.size
            start = (conv_full.size - len_x) // 2
            end = start + len_x
            result = conv_full[start:end]
        elif mode == "valid":
            len_x, len_y = x.size, y.size
            if len_x >= len_y:
                valid_len = len_x - len_y + 1
                result = conv_full[:valid_len]
            else:
                result = np.empty(0, dtype=float)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return {"convolution": result.tolist()}


def run_solver(problem: dict) -> dict:
    """
    Entry point used by the evaluator.
    """
    solver = FFTConvolution()
    return solver.solve(problem)