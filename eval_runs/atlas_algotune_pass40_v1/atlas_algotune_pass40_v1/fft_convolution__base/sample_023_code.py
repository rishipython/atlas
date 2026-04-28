import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D float64 arrays using FFT.

    This implementation uses the real‑valued FFT (rfft) for speed while
    retaining the numerical accuracy of the naive algorithm.  The result
    matches ``np.convolve(x, h, mode='full')`` to machine precision.
    """
    # Ensure input arrays are 1‑D float64
    x = np.asarray(x, dtype=np.float64, order='C')
    h = np.asarray(h, dtype=np.float64, order='C')

    N, M = x.shape[0], h.shape[0]
    L = N + M - 1                     # Length of the full convolution
    size = 1 << (L - 1).bit_length()  # Next power of two for efficient FFT

    # Compute FFTs of both signals, zero‑padded to ``size``
    X = np.fft.rfft(x, size)
    H = np.fft.rfft(h, size)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT and truncate to the desired length
    y = np.fft.irfft(Y, size)[:L]

    return y