import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays using FFT.
    Equivalent to np.convolve(x, h, mode='full') but faster for large inputs.
    """
    # Ensure inputs are 1-D and float64
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N = x.shape[0]
    M = h.shape[0]
    L = N + M - 1  # length of the full convolution result

    # Use real FFT (rfft) for efficiency, since inputs are real
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Element-wise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT to get back to time domain
    y = np.fft.irfft(Y, n=L)

    return y