import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT."""
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    # Length of the full convolution result
    L = x.size + h.size - 1

    # For very small inputs, the overhead of FFT outweighs the benefit;
    # use the reference algorithm in that case.
    if L <= 64:
        out = np.zeros(L, dtype=np.float64)
        for n in range(L):
            kmin = max(0, n - h.size + 1)
            kmax = min(x.size - 1, n)
            out[n] = np.sum(x[kmin : kmax + 1] * h[n - np.arange(kmin, kmax + 1)])
        return out

    # FFT‑based convolution using real‑to‑complex transforms
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)
    y = np.fft.irfft(X * H, n=L)

    return y