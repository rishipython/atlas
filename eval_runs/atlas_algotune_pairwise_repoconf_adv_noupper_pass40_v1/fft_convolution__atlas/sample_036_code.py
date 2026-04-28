import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D float64 arrays
    using an FFT‑based approach for speed.
    """
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N = x.size
    M = h.size
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of the full convolution result
    out_len = N + M - 1

    # Next power of two for efficient FFT
    L = 1 << (out_len - 1).bit_length()

    # FFTs of the padded inputs
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get the convolution result
    y_full = np.fft.irfft(Y, n=L, dtype=np.float64)[:out_len]

    return y_full