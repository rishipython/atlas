import numpy as np
from scipy import signal
import logging
from typing import Dict, Any


class FFTConvolution:
    """
    Optimised FFT convolution implementation.
    """

    def __init__(self, small_threshold: int = 1024):
        """
        Initialise the solver.

        Parameters
        ----------
        small_threshold : int
            For signals whose product of lengths is below this threshold
            we use the direct convolution implementation from numpy.
        """
        self.small_threshold = small_threshold

    @staticmethod
    def _next_fast_len(n: int) -> int:
        """
        Return the next fast length for FFT from numpy.

        Parameters
        ----------
        n : int
            Desired length.

        Returns
        -------
        int
            Length suitable for fast FFT.
        """
        try:
            return np.fft.next_fast_len(n)
        except AttributeError:
            # Fallback for older numpy versions
            return 2 ** int(np.ceil(np.log2(n)))

    def _fft_convolve(self, x: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
        """
        Perform convolution using FFT.

        Parameters
        ----------
        x : np.ndarray
            First input signal.
        y : np.ndarray
            Second input signal.
        mode : str
            Convolution mode: 'full', 'same', or 'valid'.

        Returns
        -------
        np.ndarray
            Convolution result in the requested mode.
        """
        len_x, len_y = len(x), len(y)

        # Handle empty input
        if len_x == 0 or len_y == 0:
            return np.array([], dtype=float)

        # Decide whether to use direct convolution
        if len_x * len_y <= self.small_threshold:
            return np.convolve(x, y, mode=mode)

        # Full convolution length
        full_len = len_x + len_y - 1

        # Pad to next fast length
        n_fft = self._next_fast_len(full_len)

        # Real FFT for real input signals
        fx = np.fft.rfft(x, n=n_fft)
        fy = np.fft.rfft(y, n=n_fft)

        # Element‑wise multiplication in frequency domain
        fz = fx * fy

        # Inverse real FFT
        z = np.fft.irfft(fz, n=n_fft)

        # Truncate to the true full convolution length
        z = z[:full_len]

        # Slice according to mode
        if mode == "full":
            return z
        elif mode == "same":
            start = (full_len - len_x) // 2
            return z[start:start + len_x]
        elif mode == "valid":
            if len_x >= len_y:
                start = len_y - 1
                length = len_x - len_y + 1
            else:
                start = len_x - 1
                length = len_y - len_x + 1
            return z[start:start + length]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the FFT convolution problem.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - "signal_x": list of numbers
                - "signal_y": list of numbers
                - "mode": "full", "same", or "valid"

        Returns
        -------
        dict
            Dictionary with key "convolution" containing the result list.
        """
        try:
            signal_x = np.asarray(problem["signal_x"], dtype=float)
            signal_y = np.asarray(problem["signal_y"], dtype=float)
            mode = problem.get("mode", "full")

            result = self._fft_convolve(signal_x, signal_y, mode)

            return {"convolution": result.tolist()}
        except Exception as e:
            logging.error(f"Error in solve method: {e}")
            raise


def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for the evaluator.

    Parameters
    ----------
    problem : dict
        Problem dictionary as described in the task.

    Returns
    -------
    dict
        Solver output.
    """
    solver = FFTConvolution()
    return solver.solve(problem)