import numpy as np
try:
    from scipy.signal import fftconvolve
except Exception:  # pragma: no cover
    fftconvolve = None

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D float64 arrays.

    The implementation uses `scipy.signal.fftconvolve` if available,
    otherwise falls back to a NumPy‑based FFT approach.  The result
    matches `np.convolve(x, h, mode='full')` to machine precision.
    """
    # Ensure we are working with float64 arrays
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    # Handle empty inputs explicitly
    if x.size == 0 or h.size == 0:
        return np.empty(0, dtype=np.float64)

    if fftconvolve is not None:
        # SciPy's implementation is highly optimized
        return fftconvolve(x, h, mode="full")

    # Fallback: use NumPy FFT directly
    # Compute next power of two for efficient FFT
    n = x.size + h.size - 1
    fft_size = 1 << (n - 1).bit_length()
    X = np.fft.rfft(x, n=fft_size)
    H = np.fft.rfft(h, n=fft_size)
    y = np.fft.irfft(X * H, n=fft_size)
    return y[:n]