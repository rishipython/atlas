import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays x and h
    using an FFT-based algorithm for speed. The result is numerically
    close to that of np.convolve(x, h, mode='full').
    """
    # Ensure inputs are 1-D and of type float64
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.size, h.size
    conv_len = N + M - 1

    if conv_len == 0:
        return np.empty(0, dtype=np.float64)

    # Choose an efficient FFT length
    L = np.fft.next_fast_len(conv_len)

    # Compute FFTs
    X = np.fft.fft(x, n=L)
    H = np.fft.fft(h, n=L)

    # Pointwise multiplication and inverse FFT
    y = np.fft.ifft(X * H)

    # Return the real part of the convolution, truncated to the required length
    return np.real(y[:conv_len])