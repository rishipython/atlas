import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays using FFT.
    This implementation is numerically equivalent to np.convolve(x, h, mode='full')
    but is considerably faster for large inputs.
    """
    # Ensure inputs are 1-D numpy arrays of type float64
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.size, h.size
    # Handle empty inputs
    if N == 0 or M == 0:
        return np.empty(0, dtype=np.float64)

    out_len = N + M - 1

    # Choose FFT length: next power of two >= out_len for speed
    n_fft = 1 << (out_len - 1).bit_length()

    # Compute forward FFTs using real FFT for efficiency
    X = np.fft.rfft(x, n_fft)
    H = np.fft.rfft(h, n_fft)

    # Element-wise product in frequency domain and inverse FFT
    y = np.fft.irfft(X * H, n_fft)

    # Truncate to the true convolution length
    return y[:out_len]