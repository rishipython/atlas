import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays.
    This implementation uses the real FFT for speed.
    """
    N, M = x.shape[0], h.shape[0]
    L = N + M - 1

    # Compute the real FFT of both sequences padded to length L
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT to obtain the convolution result
    y = np.fft.irfft(Y, n=L)

    # Ensure the result is of type float64
    return y.astype(np.float64)