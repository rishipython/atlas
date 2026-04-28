import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D float64 arrays using FFT."""
    # Ensure inputs are 1‑D float64
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.size, h.size
    # Length of the result
    out_len = N + M - 1

    if out_len == 0:  # one of the inputs is empty
        return np.empty(0, dtype=np.float64)

    # Compute the next power of two for efficient FFT
    L = 1 << (out_len - 1).bit_length()

    # FFT of both signals (real FFT)
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT to get the convolution result
    y = np.fft.irfft(Y, n=L)

    # Return the linear convolution length
    return y[:out_len]