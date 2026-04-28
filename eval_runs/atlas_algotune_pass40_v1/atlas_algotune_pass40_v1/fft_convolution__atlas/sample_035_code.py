import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays using FFT.
    Equivalent to np.convolve(x, h, mode='full') but faster for large inputs.
    """
    # Ensure input arrays are 1-D and of type float64
    x = np.asarray(x, dtype=np.float64, order="C")
    h = np.asarray(h, dtype=np.float64, order="C")
    if x.ndim != 1 or h.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays")

    N = x.shape[0]
    M = h.shape[0]
    out_len = N + M - 1

    # Next power of two for efficient FFT (optional but speeds up large transforms)
    L = 1 << (out_len - 1).bit_length()

    # Compute FFTs of zero-padded inputs using real FFT for speed
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Element-wise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT to get convolution result
    y_full = np.fft.irfft(Y, n=L)

    # Truncate to the exact linear convolution length
    return y_full[:out_len]