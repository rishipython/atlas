import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return np.convolve(x, h, mode='full') but MUCH faster using FFT."""
    # Ensure inputs are 1-D and cast to float64
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N = x.size
    M = h.size
    L = N + M - 1  # length of the full convolution

    # Compute FFTs of appropriate length
    X = np.fft.fft(x, n=L)
    H = np.fft.fft(h, n=L)

    # Element‑wise multiplication in frequency domain and inverse FFT
    y = np.fft.ifft(X * H).real

    # Ensure the result is float64
    return y.astype(np.float64)