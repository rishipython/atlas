import numpy as np
from numpy.fft import rfft, irfft, next_fast_len
import logging

class FFTConvolution:
    """
    Optimized FFT convolution solver.
    Uses numpy's FFT routines and falls back to direct convolution for small inputs.
    """

    def __init__(self):
        pass

    def solve(self, problem):
        """
        Solve the convolution problem using FFT or direct convolution for small inputs.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                "signal_x" : list[float]
                "signal_y" : list[float]
                "mode"     : str, one of "full", "same", "valid"

        Returns
        -------
        dict
            {"convolution": list[float]}
        """
        try:
            signal_x = np.asarray(problem["signal_x"], dtype=np.float64)
            signal_y = np.asarray(problem["signal_y"], dtype=np.float64)
            mode = problem.get("mode", "full").lower()

            # Handle empty inputs
            if signal_x.size == 0 or signal_y.size == 0:
                return {"convolution": []}

            # Use direct convolution for very small inputs
            if signal_x.size * signal_y.size < 5000:
                full = np.convolve(signal_x, signal_y)
                return {"convolution": self._slice_mode(full, mode, signal_x.size, signal_y.size)}

            # Linear convolution length
            full_len = signal_x.size + signal_y.size - 1
            n = next_fast_len(full_len)

            # FFT of zero‑padded signals (real FFT)
            X = rfft(signal_x, n)
            Y = rfft(signal_y, n)
            Z = irfft(X * Y, n)

            # Keep only the linear part
            Z = Z[:full_len]

            result = self._slice_mode(Z, mode, signal_x.size, signal_y.size)
            return {"convolution": result.tolist()}

        except Exception as e:
            logging.error(f"Error in solve method: {e}")
            raise

    @staticmethod
    def _slice_mode(full, mode, len_x, len_y):
        """
        Slice the full convolution array according to the requested mode.

        Parameters
        ----------
        full : np.ndarray
            Full linear convolution.
        mode : str
            "full", "same", or "valid".
        len_x, len_y : int
            Lengths of the input signals.

        Returns
        -------
        np.ndarray
            The sliced convolution.
        """
        if mode == "full":
            return full
        if mode == "same":
            out_len = max(len_x, len_y)
            start = (full.size - out_len) // 2
            return full[start:start + out_len]
        if mode == "valid":
            if len_x >= len_y:
                start = len_y - 1
                out_len = len_x - len_y + 1
            else:
                start = len_x - 1
                out_len = len_y - len_x + 1
            return full[start:start + out_len]
        raise ValueError(f"Unsupported mode: {mode}")

def run_solver(problem):
    """
    Entry point for the evaluator.

    Parameters
    ----------
    problem : dict
        Problem definition as described in the task.

    Returns
    -------
    dict
        The solution dictionary containing the convolution result.
    """
    solver = FFTConvolution()
    return solver.solve(problem)