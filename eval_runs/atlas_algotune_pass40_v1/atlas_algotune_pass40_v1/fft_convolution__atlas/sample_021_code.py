import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two real 1‑D float64 arrays
    using an FFT based algorithm.  The result is numerically
    indistinguishable from ``np.convolve(x, h, mode='full')``.
    """
    x = np.asarray(x, dtype=np.float64, copy=False)
    h = np.asarray(h, dtype=np.float64, copy=False)

    # Handle empty inputs
    if x.size == 0 or h.size == 0:
        return np.empty(0, dtype=np.float64)

    N, M = x.size, h.size
    conv_len = N + M - 1

    # Choose FFT length: next power of two >= conv_len
    nfft = 1 << (conv_len - 1).bit_length()

    # Compute FFTs of zero‑padded inputs
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Pointwise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to time domain and truncate to required length
    y = np.fft.irfft(Y, nfft)
    return y[:conv_len]