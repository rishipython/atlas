import numpy as np
import logging
from typing import Dict, Any

class FFTConvolution:
    """
    Fast FFT-based convolution solver with support for 'full', 'same', and 'valid' modes.
    Uses numpy's real FFT for speed and falls back to direct convolution for very small inputs.
    """

    def __init__(self):
        pass

    @staticmethod
    def _fft_convolve(signal_x: np.ndarray, signal_y: np.ndarray, mode: str):
        """
        Perform convolution using FFT (real-valued) and slice according to mode.
        """
        len_x, len_y = len(signal_x), len(signal_y)
        if len_x == 0 or len_y == 0:
            return np.array([], dtype=float)

        # Full convolution length
        L = len_x + len_y - 1

        # Next power of two for efficient FFT
        n = 1 << (L - 1).bit_length()

        # FFT of both signals
        fft_x = np.fft.rfft(signal_x, n)
        fft_y = np.fft.rfft(signal_y, n)

        # Element-wise multiplication in frequency domain
        prod = fft_x * fft_y

        # Inverse FFT to get linear convolution
        conv_full = np.fft.irfft(prod, n)[:L]

        if mode == "full":
            return conv_full
        elif mode == "same":
            # Center the output to length len_x
            offset = (len_y - 1) // 2
            start = offset
            end = start + len_x
            return conv_full[start:end]
        elif mode == "valid":
            # Number of valid points
            if len_x >= len_y:
                start = len_y - 1
                length = len_x - len_y + 1
            else:
                start = len_x - 1
                length = len_y - len_x + 1
            return conv_full[start:start + length]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    @staticmethod
    def _direct_convolve(signal_x: np.ndarray, signal_y: np.ndarray, mode: str):
        """
        Direct convolution using numpy.convolve for small inputs.
        """
        return np.convolve(signal_x, signal_y, mode=mode)

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the FFT convolution problem.
        """
        try:
            signal_x = np.asarray(problem["signal_x"], dtype=float)
            signal_y = np.asarray(problem["signal_y"], dtype=float)
            mode = problem.get("mode", "full")

            # If signals are very short, use direct convolution to avoid FFT overhead
            if signal_x.size * signal_y.size <= 5000:
                conv = self._direct_convolve(signal_x, signal_y, mode)
            else:
                conv = self._fft_convolve(signal_x, signal_y, mode)

            return {"convolution": conv.tolist()}
        except Exception as e:
            logging.error(f"Error in solve method: {e}")
            raise

    def is_solution(self, problem: Dict[str, Any], solution: Dict[str, Any]) -> bool:
        """
        Validate the convolution result.
        """
        try:
            if "convolution" not in solution:
                logging.error("Solution missing 'convolution' key.")
                return False

            student_result = solution["convolution"]

            if not isinstance(student_result, list):
                logging.error("Convolution result must be a list.")
                return False

            try:
                student_arr = np.array(student_result, dtype=float)
                if not np.all(np.isfinite(student_arr)):
                    logging.error("Convolution result contains non-finite values.")
                    return False
            except ValueError:
                logging.error("Could not convert convolution result to a numeric array.")
                return False

            signal_x = np.array(problem["signal_x"])
            signal_y = np.array(problem["signal_y"])
            mode = problem.get("mode", "full")

            len_x, len_y = len(signal_x), len(signal_y)
            if mode == "full":
                expected_len = len_x + len_y - 1
            elif mode == "same":
                expected_len = len_x
            elif mode == "valid":
                expected_len = max(0, abs(len_x - len_y) + 1)
            else:
                logging.error(f"Invalid mode: {mode}")
                return False

            if len_x == 0 or len_y == 0:
                expected_len = 0

            if student_arr.size != expected_len:
                logging.error(
                    f"Incorrect result length for mode '{mode}'. "
                    f"Expected {expected_len}, got {student_arr.size}."
                )
                return False

            # Reference solution using scipy for correctness
            from scipy import signal
            try:
                reference = signal.fftconvolve(signal_x, signal_y, mode=mode)
            except Exception as e:
                logging.error(f"Reference calculation failed: {e}")
                return False

            # Compare numerically
            if not np.allclose(student_arr, reference, rtol=1e-6, atol=1e-6):
                diff = np.abs(student_arr - reference)
                logging.error(
                    f"Numerical difference exceeds tolerance. "
                    f"Max diff: {diff.max():.2e}"
                )
                return False

            return True
        except Exception as e:
            logging.error(f"Error in is_solution method: {e}")
            return False


def run_solver(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for the solver.
    """
    solver = FFTConvolution()
    return solver.solve(problem)