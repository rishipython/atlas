import numpy as np
from typing import Dict, Any

class FFTConvolution:
    """
    Optimised FFT‑based convolution solver.
    """

    def __init__(self):
        pass

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the convolution of two real‑valued signals using FFT.
        The result is returned under the key 'convolution'.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - 'signal_x': list of floats
                - 'signal_y': list of floats
                - 'mode': str, one of 'full', 'same', 'valid' (default 'full')

        Returns
        -------
        dict
            {"convolution": list of floats}
        """
        # Convert inputs to contiguous float arrays
        x = np.array(problem["signal_x"], dtype=float, copy=False)
        y = np.array(problem["signal_y"], dtype=float, copy=False)
        mode = problem.get("mode", "full")

        # Handle empty inputs
        if x.size == 0 or y.size == 0:
            return {"convolution": []}

        # For very small signals use direct convolution to avoid FFT overhead
        if x.size * y.size <= 256:
            full = np.convolve(x, y)
        else:
            # Linear convolution via FFT
            n_full = x.size + y.size - 1
            # Pad to next power of two for optimal FFT performance
            m = 1 << (n_full - 1).bit_length()

            # Real‑to‑complex FFT (half‑size)
            fx = np.fft.rfft(x, n=m)
            fy = np.fft.rfft(y, n=m)

            # Point‑wise multiplication
            f = fx * fy

            # Inverse FFT and truncate to full length
            full = np.fft.irfft(f, n=m)[:n_full]

        # Slice according to mode
        if mode == "full":
            result = full
        elif mode == "same":
            # Central part of the convolution
            if x.size >= y.size:
                start = (y.size - 1) // 2
                end = start + x.size
            else:
                start = (x.size - 1) // 2
                end = start + y.size
            result = full[start:end]
        elif mode == "valid":
            # Only fully overlapping parts
            if x.size >= y.size:
                start = y.size - 1
                length = x.size - y.size + 1
            else:
                start = x.size - 1
                length = y.size - x.size + 1
            # Handle non‑overlap case
            if length <= 0:
                result = np.array([], dtype=float)
            else:
                result = full[start:start + length]
        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

        return {"convolution": result.tolist()}


def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point used by the evaluator.

    Parameters
    ----------
    problem : dict
        Problem description as expected by FFTConvolution.solve.

    Returns
    -------
    dict
        The solver's output dictionary.
    """
    solver = FFTConvolution()
    return solver.solve(problem)