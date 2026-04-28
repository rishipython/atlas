import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays using FFT.
    Equivalent to np.convolve(x, h, mode='full') but much faster for large inputs.
    """
    # Ensure inputs are 1-D float64
    x = np.asarray(x, dtype=np.float64, order='C').ravel()
    h = np.asarray(h, dtype=np.float64, order='C').ravel()

    N = x.shape[0]
    M = h.shape[0]
    L = N + M - 1  # length of the full convolution result

    # Use the next power of two for efficient FFT (optional but speeds up)
    nfft = 1 << (L - 1).bit_length()

    # Compute FFTs of zero-padded inputs
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Element-wise multiplication in frequency domain and inverse FFT
    y = np.fft.irfft(X * H, nfft)

    # Truncate to the required length
    return y[:L]