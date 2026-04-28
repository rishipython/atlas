import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays using FFT.
    The result is equivalent to np.convolve(x, h, mode='full').
    """
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    # Handle empty inputs like numpy's convolve
    if x.size == 0 or h.size == 0:
        return np.array([], dtype=np.float64)

    N, M = x.size, h.size
    out_len = N + M - 1

    # Next power of two for efficient FFT
    nfft = 1 << (out_len - 1).bit_length()

    # FFTs of the zero-padded inputs
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Pointwise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get convolution result
    y = np.fft.irfft(Y, nfft)

    # Truncate to the exact convolution length
    return y[:out_len]