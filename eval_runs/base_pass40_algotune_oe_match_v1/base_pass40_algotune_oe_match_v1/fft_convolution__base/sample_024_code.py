import numpy as np
from scipy import signal

class FFTConvolution:
    """
    Optimized FFT convolution solver.
    Uses real FFT (rfft/irfft) and minimal padding for speed.
    """

    def __init__(self):
        pass

    def _next_pow2(self, n: int) -> int:
        """Return the next power of two >= n."""
        return 1 << (n - 1).bit_length()

    def solve(self, problem):
        """
        Compute convolution of two real signals using FFT.
        Supports modes: 'full', 'same', 'valid'.
        Returns a dictionary with key 'convolution'.
        """
        try:
            # Convert inputs to numpy arrays (float64)
            x = np.asarray(problem["signal_x"], dtype=np.float64)
            y = np.asarray(problem["signal_y"], dtype=np.float64)
            mode = problem.get("mode", "full")

            # Handle empty inputs early
            if x.size == 0 or y.size == 0:
                return {"convolution": []}

            len_x, len_y = x.size, y.size
            full_len = len_x + len_y - 1

            # Choose FFT length: next power of two >= full_len
            n_fft = self._next_pow2(full_len)

            # Real FFT of both signals padded to n_fft
            X = np.fft.rfft(x, n=n_fft)
            Y = np.fft.rfft(y, n=n_fft)

            # Element‑wise multiplication in frequency domain
            Z = X * Y

            # Inverse real FFT to get linear convolution
            conv_full = np.fft.irfft(Z, n=n_fft).real
            conv_full = conv_full[:full_len]  # truncate to exact length

            # Slice according to mode
            if mode == "full":
                result = conv_full
            elif mode == "same":
                out_len = max(len_x, len_y)
                start = (full_len - out_len) // 2
                result = conv_full[start:start + out_len]
            elif mode == "valid":
                # valid length = abs(len_x - len_y) + 1, but zero if any empty
                out_len = abs(len_x - len_y) + 1
                start = min(len_x, len_y) - 1
                result = conv_full[start:start + out_len]
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            return {"convolution": result.tolist()}

        except Exception as e:
            raise RuntimeError(f"Error in FFTConvolution.solve: {e}") from e


def run_solver(problem):
    """
    Entry point for the evaluator.
    """
    solver = FFTConvolution()
    return solver.solve(problem)