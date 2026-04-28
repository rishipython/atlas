import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    # Handle empty inputs: np.convolve([] , ...) returns an empty array
    if x.size == 0 or h.size == 0:
        return np.empty(0, dtype=np.float64)

    # Length of the full convolution result
    n = x.size + h.size - 1

    # Choose FFT size: next power of two >= n for efficiency
    Nfft = 1 << (n - 1).bit_length()

    # FFT of both signals (real FFT is sufficient for real inputs)
    X = np.fft.rfft(x, n=Nfft)
    H = np.fft.rfft(h, n=Nfft)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get back to time domain
    y = np.fft.irfft(Y, n=Nfft)

    # Truncate to the exact convolution length
    return y[:n]