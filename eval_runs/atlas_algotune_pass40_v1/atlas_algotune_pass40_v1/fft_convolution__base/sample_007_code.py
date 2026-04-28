import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT."""
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64, order='C')
    h = np.asarray(h, dtype=np.float64, order='C')
    N, M = x.shape[0], h.shape[0]
    out_len = N + M - 1

    # If either array is empty, return an empty array
    if N == 0 or M == 0:
        return np.empty(out_len, dtype=np.float64)

    # Determine FFT length: next power of two >= out_len
    nfft = 1 << (out_len - 1).bit_length()

    # Compute FFTs using real FFT for efficiency
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Element-wise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT to get convolution result
    y = np.fft.irfft(Y, nfft)[:out_len]

    return y