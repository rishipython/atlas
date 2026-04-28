import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT."""
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    # Handle empty inputs
    if x.size == 0 or h.size == 0:
        return np.array([], dtype=np.float64)

    # Length of the full convolution result
    out_len = x.size + h.size - 1

    # Next power of two for efficient FFT (optional but speeds up large cases)
    fft_len = 1 << (out_len - 1).bit_length()

    # Compute FFTs of zero‑padded inputs
    X = np.fft.rfft(x, fft_len)
    H = np.fft.rfft(h, fft_len)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get convolution result
    y = np.fft.irfft(Y, fft_len)

    # Truncate to the exact output length
    return y[:out_len]