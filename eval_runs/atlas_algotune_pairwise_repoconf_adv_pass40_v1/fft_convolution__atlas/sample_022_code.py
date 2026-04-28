import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1‑D float64 arrays.
    Uses an FFT‑based algorithm for speed while preserving numerical
    accuracy comparable to np.convolve(..., mode='full').
    """
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.size, h.size
    out_len = N + M - 1

    # For very short inputs, the direct algorithm is faster
    if out_len <= 64:
        out = np.zeros(out_len, dtype=np.float64)
        for n in range(out_len):
            kmin = max(0, n - M + 1)
            kmax = min(N - 1, n)
            out[n] = np.dot(x[kmin:kmax + 1], h[n - kmax:n - kmin + 1][::-1])
        return out

    # FFT‑based convolution
    # Use real FFT for efficiency (inputs are real)
    nfft = out_len
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)
    Y = X * H
    y = np.fft.irfft(Y, nfft)

    # The result may have a tiny imaginary part due to numerical errors
    return y.astype(np.float64)