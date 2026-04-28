import numpy as np
import logging
from typing import Dict, Any

class FFTConvolution:
    """
    Optimised FFT convolution implementation using NumPy's real FFT.
    """

    @staticmethod
    def _fft_convolve_real(x: np.ndarray, y: np.ndarray, mode: str = "full") -> np.ndarray:
        """
        Compute the linear convolution of two real signals using FFT.

        Parameters
        ----------
        x, y : np.ndarray
            1‑D real input signals.
        mode : str
            One of ``"full"``, ``"same"``, ``"valid"``.  The result is sliced
            from the full convolution accordingly.

        Returns
        -------
        np.ndarray
            Convolution result in the requested mode.
        """
        # Handle empty inputs early
        if x.size == 0 or y.size == 0:
            return np.array([], dtype=np.float64)

        # Length of the full convolution
        n_full = x.size + y.size - 1

        # Use real FFT to avoid unnecessary complex arithmetic
        n_fft = n_full
        X = np.fft.rfft(x, n=n_fft)
        Y = np.fft.rfft(y, n=n_fft)
        prod = X * Y
        full = np.fft.irfft(prod, n=n_fft)

        if mode == "full":
            return full

        # Determine slice indices for "same" and "valid"
        len_x, len_y = x.size, y.size
        if mode == "same":
            # Center part of length max(len_x, len_y)
            out_len = max(len_x, len_y)
            start = (n_full - out_len) // 2
            return full[start:start + out_len]
        elif mode == "valid":
            # Only where full overlap occurs
            if len_x >= len_y:
                start = len_y - 1
                out_len = len_x - len_y + 1
            else:
                start = len_x - 1
                out_len = len_y - len_x + 1
            return full[start:start + out_len]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the FFT convolution problem.

        Parameters
        ----------
        problem : dict
            Must contain keys:
            - "signal_x": list or array of floats
            - "signal_y": list or array of floats
            - "mode": optional, one of "full", "same", "valid" (default "full")

        Returns
        -------
        dict
            {"convolution": list of floats}
        """
        try:
            # Convert inputs to NumPy arrays of type float64
            x = np.asarray(problem["signal_x"], dtype=np.float64)
            y = np.asarray(problem["signal_y"], dtype=np.float64)
            mode = problem.get("mode", "full")

            # Validate mode
            if mode not in ("full", "same", "valid"):
                raise ValueError(f"Invalid mode '{mode}'. Expected 'full', 'same', or 'valid'.")

            # Compute convolution
            conv = self._fft_convolve_real(x, y, mode=mode)

            # Convert to list for JSON serialisation
            return {"convolution": conv.tolist()}
        except Exception as exc:
            logging.exception("Error in FFTConvolution.solve")
            raise exc

    def is_solution(self, problem: Dict[str, Any], solution: Dict[str, Any]) -> bool:
        """
        Validate the solution against the reference implementation.
        """
        try:
            import scipy.signal

            # Basic key and type checks
            if "convolution" not in solution:
                logging.error("Solution missing 'convolution' key.")
                return False
            if not isinstance(solution["convolution"], list):
                logging.error("Convolution result must be a list.")
                return False

            # Convert to NumPy array
            student = np.array(solution["convolution"], dtype=np.float64)
            if not np.all(np.isfinite(student)):
                logging.error("Convolution result contains non-finite values.")
                return False

            # Reference solution
            ref = scipy.signal.fftconvolve(
                np.asarray(problem["signal_x"], dtype=np.float64),
                np.asarray(problem["signal_y"], dtype=np.float64),
                mode=problem.get("mode", "full")
            )

            # Compare with tolerance
            if not np.allclose(student, ref, rtol=1e-6, atol=1e-6):
                diff = np.abs(student - ref)
                logging.error(
                    f"Numerical difference exceeds tolerance. "
                    f"Max diff: {diff.max():.2e}, Avg diff: {diff.mean():.2e}"
                )
                return False

            return True
        except Exception as exc:
            logging.exception("Error in FFTConvolution.is_solution")
            return False

def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point used by the evaluation harness.

    Parameters
    ----------
    problem : dict
        Problem definition as described in the task.

    Returns
    -------
    dict
        Solution dictionary containing the convolution result.
    """
    solver = FFTConvolution()
    return solver.solve(problem)