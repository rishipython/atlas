import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure inputs are 1‑D and of float64 dtype
    x = np.asarray(x, dtype=np.float64, copy=False)
    h = np.asarray(h, dtype=np.float64, copy=False)
    N = x.shape[0]
    M = h.shape[0]
    out_len = N + M - 1

    # Use the next power of two for efficient FFT
    fft_len = 1 << (out_len - 1).bit_length()

    # Compute FFTs of both signals padded to fft_len
    X = np.fft.rfft(x, fft_len)
    H = np.fft.rfft(h, fft_len)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get back to time domain
    y = np.fft.irfft(Y, fft_len)

    # Truncate to the exact convolution length
    return y[:out_len]