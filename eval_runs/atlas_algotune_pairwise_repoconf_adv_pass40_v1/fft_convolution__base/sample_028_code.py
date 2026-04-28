import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D float64 arrays
    using an FFT‑based approach, matching np.convolve(x, h, 'full')."""
    # Ensure input is 1‑D
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.shape[0], h.shape[0]
    out_len = N + M - 1

    # Handle trivial cases
    if out_len <= 0:
        return np.array([], dtype=np.float64)

    # Choose FFT length as next power of two for speed
    nfft = 1 << (out_len - 1).bit_length()

    # Perform real‑to‑complex FFTs
    X = np.fft.rfft(x, n=nfft)
    H = np.fft.rfft(h, n=nfft)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get convolution result
    y = np.fft.irfft(Y, n=nfft)

    # Truncate to the exact output length
    return y[:out_len]