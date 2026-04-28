import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays
    using an FFT-based algorithm for speed.
    """
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64, order='C')
    h = np.asarray(h, dtype=np.float64, order='C')
    N, M = x.size, h.size
    L = N + M - 1

    # Next power of two for efficient FFT
    nfft = 1 << (L - 1).bit_length()

    # FFT of zero-padded inputs
    X = np.fft.fft(x, nfft)
    H = np.fft.fft(h, nfft)

    # Element-wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT and take real part
    y = np.fft.ifft(Y).real

    # Truncate to the exact convolution length
    return y[:L]