import numpy as np
from numpy.fft import rfft, irfft

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two real 1‑D float64 arrays
    using an FFT‑based method. Equivalent to np.convolve(x, h, mode='full')
    but far faster for large inputs.
    """
    # Ensure inputs are real float64 arrays
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N = x.shape[0]
    M = h.shape[0]

    # Handle empty inputs
    if N == 0 or M == 0:
        return np.empty((0,), dtype=np.float64)

    out_len = N + M - 1
    # Choose FFT length that is a power‑of‑two or otherwise efficient
    L = np.fft.next_fast_len(out_len)

    # Compute FFTs
    X = rfft(x, n=L)
    H = rfft(h, n=L)

    # Element‑wise multiplication and inverse FFT
    Y = irfft(X * H, n=L)

    # Truncate to the exact convolution length
    return Y[:out_len]