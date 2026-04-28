import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1‑D float64 arrays
    using an FFT‑based method for speed.
    Equivalent to np.convolve(x, h, mode='full').
    """
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64, copy=False)
    h = np.asarray(h, dtype=np.float64, copy=False)

    N, M = x.shape[0], h.shape[0]
    out_len = N + M - 1

    # Zero‑pad to next power of two for efficient FFT
    nfft = 1 << (out_len - 1).bit_length()

    # Compute FFTs using real FFT for speed
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT to get the convolution result
    y = np.fft.irfft(Y, nfft)

    # Truncate to the exact output length
    return y[:out_len].astype(np.float64)