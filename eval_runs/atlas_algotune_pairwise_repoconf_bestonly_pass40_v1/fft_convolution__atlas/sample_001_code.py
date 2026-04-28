import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure input is 1‑D and float64
    x = np.asarray(x, dtype=np.float64, order='C')
    h = np.asarray(h, dtype=np.float64, order='C')
    N = x.shape[0]
    M = h.shape[0]
    L = N + M - 1

    # Choose FFT length as next power of two for speed
    fftlen = 1 << (L - 1).bit_length()

    # Compute FFTs
    X = np.fft.fft(x, n=fftlen)
    H = np.fft.fft(h, n=fftlen)

    # Point‑wise multiplication and inverse FFT
    y = np.fft.ifft(X * H).real[:L]

    return y.astype(np.float64)