import logging
import numpy as np
from scipy import signal
from typing import Dict, Any

class FFTConvolution:
    """
    Optimized FFT convolution solver.
    Uses NumPy's real FFT for speed and handles all requested modes.
    """

    def __init__(self):
        pass

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the convolution of two real signals using FFT.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - "signal_x": list of floats
                - "signal_y": list of floats
                - "mode": "full" | "same" | "valid"

        Returns
        -------
        dict
            {"convolution": list of floats}
        """
        try:
            # Extract signals and mode
            x = np.asarray(problem.get("signal_x", []), dtype=float)
            y = np.asarray(problem.get("signal_y", []), dtype=float)
            mode = problem.get("mode", "full")

            len_x, len_y = x.size, y.size

            # Handle empty inputs
            if len_x == 0 or len_y == 0:
                return {"convolution": []}

            # Length of the full linear convolution
            full_len = len_x + len_y - 1

            # Choose FFT size: next power of two >= full_len
            nfft = 1 << (full_len - 1).bit_length()

            # Compute FFTs (real-valued signals)
            X = np.fft.rfft(x, n=nfft)
            Y = np.fft.rfft(y, n=nfft)

            # Element‑wise multiplication in frequency domain
            Z = X * Y

            # Inverse FFT to obtain linear convolution
            conv_full = np.fft.irfft(Z, n=nfft)[:full_len]

            # Slice according to the requested mode
            if mode == "full":
                result = conv_full
            elif mode == "same":
                # Output length is the larger of the two input lengths
                max_len = max(len_x, len_y)
                start = (full_len - max_len) // 2
                result = conv_full[start:start + max_len]
            elif mode == "valid":
                # Output length where signals fully overlap
                valid_len = max(len_x, len_y) - min(len_x, len_y) + 1
                start = min(len_x, len_y) - 1
                result = conv_full[start:start + valid_len]
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            return {"convolution": result.tolist()}

        except Exception as exc:
            logging.error(f"Error in FFTConvolution.solve: {exc}")
            raise


def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for the evaluator.

    Parameters
    ----------
    problem : dict
        Problem definition for the FFT convolution task.

    Returns
    -------
    dict
        Result of the convolution under the key "convolution".
    """
    solver = FFTConvolution()
    return solver.solve(problem)