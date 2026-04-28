import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays using FFT.
    Equivalent to np.convolve(x, h, mode='full') but faster for large inputs.
    """
    # Ensure inputs are 1-D and of type float64
    x = np.asarray(x, dtype=np.float64, copy=False).ravel()
    h = np.asarray(h, dtype=np.float64, copy=False).ravel()

    N = x.size
    M = h.size
    L = N + M - 1

    # Choose FFT length: next power of two for speed (optional)
    fft_len = 1 << (L - 1).bit_length()

    # Compute FFTs
    X = np.fft.fft(x, n=fft_len)
    H = np.fft.fft(h, n=fft_len)

    # Element-wise multiplication and inverse FFT
    y_full = np.fft.ifft(X * H).real

    # Truncate to the required length
    return y_full[:L]