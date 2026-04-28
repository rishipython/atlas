import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return np.convolve(x, h, mode='full') but MUCH faster using FFT."""
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N = x.size
    M = h.size
    out_len = N + M - 1

    # Use the next power of two for efficient FFT (optional but usually faster)
    fft_size = 1 << (out_len - 1).bit_length()

    # Compute FFTs using real FFT (rfft) for speed and memory efficiency
    X = np.fft.rfft(x, fft_size)
    H = np.fft.rfft(h, fft_size)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT and trim to the exact output length
    y = np.fft.irfft(Y, fft_size)[:out_len]

    # Ensure the output dtype is float64
    return y.astype(np.float64)