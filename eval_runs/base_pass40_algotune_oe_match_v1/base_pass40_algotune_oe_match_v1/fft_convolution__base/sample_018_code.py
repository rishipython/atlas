import numpy as np
from scipy import signal
import logging

# Numpy's next_fast_len is available for both FFT and rFFT
next_fast_len = np.fft.next_fast_len


class FFTConvolution:
    """
    Optimised FFT convolution solver.
    """

    def __init__(self):
        pass

    @staticmethod
    def _direct_convolve(x: np.ndarray, y: np.ndarray, mode: str):
        """Fallback direct convolution for small signals."""
        if mode == "full":
            return np.convolve(x, y, mode="full")
        elif mode == "same":
            return np.convolve(x, y, mode="same")
        elif mode == "valid":
            return np.convolve(x, y, mode="valid")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def solve(self, problem):
        """
        Compute convolution using FFT with optimisations:
        - Small signals use direct convolution.
        - rFFT is used for real-valued inputs.
        - Zero padding is minimised using next_fast_len.
        """
        try:
            # Extract inputs
            signal_x = np.asarray(problem["signal_x"], dtype=np.float64)
            signal_y = np.asarray(problem["signal_y"], dtype=np.float64)
            mode = problem.get("mode", "full")

            # Handle empty inputs
            if signal_x.size == 0 or signal_y.size == 0:
                return {"convolution": []}

            # Threshold for switching to direct convolution
            direct_threshold = 5000
            if signal_x.size * signal_y.size <= direct_threshold:
                conv = self._direct_convolve(signal_x, signal_y, mode)
                return {"convolution": conv.tolist()}

            # Linear convolution length
            len_x, len_y = signal_x.size, signal_y.size
            linear_len = len_x + len_y - 1

            # Pad to next fast length for rFFT
            fft_len = next_fast_len(linear_len)

            # Compute FFTs using rFFT (real signals)
            X = np.fft.rfft(signal_x, n=fft_len)
            Y = np.fft.rfft(signal_y, n=fft_len)

            # Element‑wise multiplication in frequency domain
            Z = X * Y

            # Inverse rFFT to get linear convolution
            conv_full = np.fft.irfft(Z, n=fft_len)[:linear_len]

            # Select the requested mode
            if mode == "full":
                result = conv_full
            elif mode == "same":
                out_len = max(len_x, len_y)
                start = (linear_len - out_len) // 2
                result = conv_full[start : start + out_len]
            elif mode == "valid":
                # Valid part length
                out_len = max(len_x, len_y) - min(len_x, len_y) + 1
                # Start index depends on which signal is longer
                start = min(len_x, len_y) - 1
                result = conv_full[start : start + out_len]
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            return {"convolution": result.tolist()}

        except Exception as e:
            logging.error(f"Error in solve method: {e}")
            raise


def run_solver(problem):
    """
    Entry point used by the evaluator.
    """
    solver = FFTConvolution()
    return solver.solve(problem)