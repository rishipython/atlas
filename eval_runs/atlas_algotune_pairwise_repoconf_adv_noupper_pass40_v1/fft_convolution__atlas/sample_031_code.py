import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure inputs are 1-D and of type float64
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N = x.shape[0]
    M = h.shape[0]
    out_len = N + M - 1

    # Compute FFTs of appropriate length
    # Using the next power of two can speed up the FFT but is not necessary.
    fft_len = out_len
    X = np.fft.fft(x, n=fft_len)
    H = np.fft.fft(h, n=fft_len)

    # Pointwise multiplication in frequency domain and inverse FFT
    y = np.fft.ifft(X * H)

    # Due to numerical errors, the result may have tiny imaginary parts.
    # Take the real part and cast to float64.
    return np.real(y).astype(np.float64)