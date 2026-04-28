import numpy as np
import logging

# --------------------------------------------------------------------------- #
# Fast FFT‑based convolution solver
# --------------------------------------------------------------------------- #
class FFTConvolution:
    """
    Optimised FFT convolution solver.

    The implementation uses real‑valued FFTs (rfft/irfft) which are
    faster than the generic FFT for real input signals.  For very small
    signals the direct convolution path is used to avoid the FFT overhead.
    The function supports the same modes as scipy.signal.fftconvolve:
        * "full"   – output length = len(x) + len(y) - 1
        * "same"   – central part of the full convolution
        * "valid"  – parts where the signals fully overlap
    """

    # Threshold below which the direct convolution is used
    _DIRECT_CONV_THRESHOLD = 500

    def __init__(self):
        pass

    @staticmethod
    def _next_pow2(n: int) -> int:
        """Return the next power of two greater than or equal to n."""
        return 1 << (n - 1).bit_length()

    @staticmethod
    def _convolve_direct(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Direct (O(n*m)) convolution using NumPy's convolve."""
        return np.convolve(x, y, mode="full")

    def _fft_convolve(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Convolve two real 1‑D signals using FFT.

        Parameters
        ----------
        x, y : np.ndarray
            Real input signals.

        Returns
        -------
        full : np.ndarray
            Full convolution result of length len(x)+len(y)-1.
        """
        n_full = x.size + y.size - 1
        n_fft = self._next_pow2(n_full)

        # Zero‑pad to FFT length and compute forward FFTs
        X = np.fft.rfft(x, n=n_fft)
        Y = np.fft.rfft(y, n=n_fft)

        # Element‑wise multiplication in frequency domain
        Z = X * Y

        # Inverse FFT and truncate to full length
        full = np.fft.irfft(Z, n=n_fft)[:n_full]
        return full

    def _slice_mode(self, full: np.ndarray, mode: str,
                    len_x: int, len_y: int) -> np.ndarray:
        """
        Extract the requested mode from the full convolution.

        Parameters
        ----------
        full : np.ndarray
            Full convolution result.
        mode : str
            One of 'full', 'same', 'valid'.
        len_x, len_y : int
            Lengths of the input signals.

        Returns
        -------
        sliced : np.ndarray
            Convolution result in the requested mode.
        """
        n_full = full.size
        if mode == "full":
            return full

        if mode == "same":
            m = max(len_x, len_y)
            start = (n_full - m) // 2
            return full[start:start + m]

        if mode == "valid":
            if len_x == 0 or len_y == 0:
                return np.empty(0, dtype=full.dtype)
            valid_len = max(0, max(len_x, len_y) - min(len_x, len_y) + 1)
            if valid_len == 0:
                return np.empty(0, dtype=full.dtype)
            start = min(len_x, len_y) - 1
            return full[start:start + valid_len]

        raise ValueError(f"Unsupported mode: {mode}")

    def solve(self, problem: dict) -> dict:
        """
        Compute the FFT‑based convolution of the input signals.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - "signal_x": list of floats
                - "signal_y": list of floats
                - "mode": str (default "full")

        Returns
        -------
        dict
            {"convolution": list of floats}
        """
        try:
            signal_x = np.asarray(problem["signal_x"], dtype=float)
            signal_y = np.asarray(problem["signal_y"], dtype=float)
            mode = problem.get("mode", "full")

            # Handle empty inputs early
            if signal_x.size == 0 or signal_y.size == 0:
                return {"convolution": []}

            # Small‑size shortcut
            if signal_x.size * signal_y.size < self._DIRECT_CONV_THRESHOLD:
                full = self._convolve_direct(signal_x, signal_y)
            else:
                full = self._fft_convolve(signal_x, signal_y)

            result = self._slice_mode(full, mode, signal_x.size, signal_y.size)

            return {"convolution": result.tolist()}

        except Exception as exc:
            logging.error(f"Error in solve: {exc}")
            raise

    def is_solution(self, problem: dict, solution: dict) -> bool:
        """
        Validate the solution against scipy's reference implementation.
        """
        from scipy import signal

        try:
            if "convolution" not in solution:
                logging.error("Solution missing 'convolution' key.")
                return False

            student = np.asarray(solution["convolution"], dtype=float)
            if not np.all(np.isfinite(student)):
                logging.error("Solution contains NaN or inf.")
                return False

            signal_x = np.asarray(problem["signal_x"], dtype=float)
            signal_y = np.asarray(problem["signal_y"], dtype=float)
            mode = problem.get("mode", "full")

            # Expected length calculation
            len_x, len_y = signal_x.size, signal_y.size
            if mode == "full":
                expected_len = len_x + len_y - 1
            elif mode == "same":
                expected_len = max(len_x, len_y)
            elif mode == "valid":
                expected_len = max(0, max(len_x, len_y) - min(len_x, len_y) + 1)
            else:
                logging.error(f"Invalid mode: {mode}")
                return False

            if len_x == 0 or len_y == 0:
                expected_len = 0

            if student.size != expected_len:
                logging.error(f"Expected length {expected_len}, got {student.size}.")
                return False

            # Reference result
            reference = signal.fftconvolve(signal_x, signal_y, mode=mode)

            # Numerical check
            if not np.allclose(student, reference, rtol=1e-6, atol=1e-6):
                diff = np.abs(student - reference)
                logging.error(
                    f"Numerical difference exceeds tolerance. "
                    f"max={diff.max():.2e}, mean={diff.mean():.2e}"
                )
                return False

            return True

        except Exception as exc:
            logging.error(f"Validation error: {exc}")
            return False


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_solver(problem: dict) -> dict:
    """
    Run the FFT convolution solver.

    Parameters
    ----------
    problem : dict
        Problem dictionary as described in the task.

    Returns
    -------
    dict
        {"convolution": list of floats}
    """
    solver = FFTConvolution()
    return solver.solve(problem)