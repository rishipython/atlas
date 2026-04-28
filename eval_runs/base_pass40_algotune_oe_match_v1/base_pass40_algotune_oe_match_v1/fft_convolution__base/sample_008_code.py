import numpy as np
import logging
from typing import Dict, Any

# --------------------------------------------------------------------------- #
# FFT Convolution solver – fast, pure NumPy implementation
# --------------------------------------------------------------------------- #

class FFTConvolution:
    """
    Fast FFT‑based convolution for 1‑D real signals.
    Supports the modes used by the reference implementation:
        - "full"   : full linear convolution
        - "same"   : central part, length = max(len(x), len(y))
        - "valid"  : fully overlapping part
    """

    def __init__(self, small_threshold: int = 1024):
        """
        Parameters
        ----------
        small_threshold : int
            For inputs whose product of lengths is below this threshold,
            the routine falls back to the direct convolution via np.convolve
            to avoid the overhead of FFT.
        """
        self.small_threshold = small_threshold

    @staticmethod
    def _next_pow_two(n: int) -> int:
        """Return the next power of two >= n."""
        return 1 << (n - 1).bit_length()

    def _fft_convolve_full(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the full linear convolution via FFT."""
        n = a.size + b.size - 1
        # If the product of lengths is small, fall back to direct convolution
        if a.size * b.size < self.small_threshold:
            return np.convolve(a, b, mode="full")

        size = self._next_pow_two(n)
        # Use real FFT for real inputs
        A = np.fft.rfft(a, size)
        B = np.fft.rfft(b, size)
        C = A * B
        conv = np.fft.irfft(C, size)[:n]
        return conv

    def _slice_full_to_mode(self, full: np.ndarray, mode: str,
                            len_a: int, len_b: int) -> np.ndarray:
        """Slice the full convolution to the requested mode."""
        if mode == "full":
            return full

        if mode == "same":
            out_len = max(len_a, len_b)
            start = (full.size - out_len) // 2
            return full[start:start + out_len]

        if mode == "valid":
            if len_a >= len_b:
                start = len_b - 1
                out_len = len_a - len_b + 1
            else:
                start = len_a - 1
                out_len = len_b - len_a + 1
            if out_len <= 0:
                return np.array([], dtype=float)
            return full[start:start + out_len]

        raise ValueError(f"Unsupported mode: {mode}")

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the convolution of the two input signals using FFT.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - "signal_x": list of floats
                - "signal_y": list of floats
                - "mode"    : one of "full", "same", "valid" (default "full")

        Returns
        -------
        dict
            {"convolution": list of floats}
        """
        try:
            # Extract and convert to NumPy arrays
            signal_x = np.asarray(problem.get("signal_x", []), dtype=np.float64)
            signal_y = np.asarray(problem.get("signal_y", []), dtype=np.float64)
            mode = problem.get("mode", "full")

            # Handle empty inputs explicitly
            if signal_x.size == 0 or signal_y.size == 0:
                return {"convolution": []}

            # Compute full convolution via FFT
            full_conv = self._fft_convolve_full(signal_x, signal_y)

            # Slice to the requested mode
            result = self._slice_full_to_mode(full_conv, mode,
                                              signal_x.size, signal_y.size)

            # Convert to Python list for consistency with the reference
            return {"convolution": result.tolist()}

        except Exception as exc:
            logging.error(f"Error in FFTConvolution.solve: {exc}")
            raise


# --------------------------------------------------------------------------- #
# Entry point used by the evaluator
# --------------------------------------------------------------------------- #

def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to run the FFT convolution solver.

    Parameters
    ----------
    problem : dict
        Problem specification as described in the task.

    Returns
    -------
    dict
        The solver output containing the key "convolution".
    """
    solver = FFTConvolution()
    return solver.solve(problem)