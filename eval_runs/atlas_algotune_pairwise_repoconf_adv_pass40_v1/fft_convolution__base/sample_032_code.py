import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64, order='C')
    h = np.asarray(h, dtype=np.float64, order='C')
    N = x.shape[0]
    M = h.shape[0]
    L = N + M - 1  # length of the linear convolution

    # Zero‑pad both inputs to length L and compute FFTs
    X = np.fft.fft(x, n=L)
    H = np.fft.fft(h, n=L)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT and take real part (imaginary part is numerical noise)
    y = np.fft.ifft(Y).real

    return y