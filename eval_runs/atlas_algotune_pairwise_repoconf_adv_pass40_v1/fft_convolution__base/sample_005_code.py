import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64, order='C')
    h = np.asarray(h, dtype=np.float64, order='C')
    N, M = x.shape[0], h.shape[0]
    L = N + M - 1  # length of the full convolution

    # Use FFT-based convolution
    X = np.fft.fft(x, n=L)
    H = np.fft.fft(h, n=L)
    y = np.fft.ifft(X * H).real  # take real part to avoid tiny imaginary noise

    return y