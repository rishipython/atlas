import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1‑D float64 arrays using an
    FFT‑based approach for speed while preserving numerical accuracy.
    """
    # Ensure input is 1‑D and of float64 type
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.size, h.size
    conv_len = N + M - 1

    # Choose an FFT length that is a power of two for speed
    fftlen = 1 << (conv_len - 1).bit_length()

    # Compute forward FFTs using real transforms
    X = np.fft.rfft(x, n=fftlen)
    H = np.fft.rfft(h, n=fftlen)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get the convolution result
    y_full = np.fft.irfft(Y, n=fftlen)

    # Return only the valid part of the convolution
    return y_full[:conv_len]