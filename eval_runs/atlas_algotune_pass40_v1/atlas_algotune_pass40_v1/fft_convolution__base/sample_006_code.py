import numpy as np

try:
    # Use SciPy's fast convolution if available
    from scipy.signal import fftconvolve
except Exception:
    # Fallback implementation using NumPy's FFT
    def fftconvolve(x: np.ndarray, h: np.ndarray, mode: str = "full") -> np.ndarray:
        n, m = x.shape[0], h.shape[0]
        size = n + m - 1
        # Next power of two for efficient FFT
        fft_len = 1 << (size - 1).bit_length()
        X = np.fft.rfft(x, fft_len)
        H = np.fft.rfft(h, fft_len)
        y = np.fft.irfft(X * H, fft_len)
        return y[:size] if mode == "full" else y  # mode handling is minimal

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return np.convolve(x, h, mode='full') but MUCH faster."""
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    return fftconvolve(x, h, mode="full")