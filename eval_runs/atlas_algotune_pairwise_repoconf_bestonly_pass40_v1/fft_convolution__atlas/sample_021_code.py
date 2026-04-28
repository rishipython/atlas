import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster."""
    # Ensure input arrays are 1-D float64
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N = x.size
    M = h.size
    L = N + M - 1

    # Handle empty inputs: np.convolve returns an empty array
    if L == 0:
        return np.array([], dtype=np.float64)

    # Choose FFT length as next power of two for speed
    nfft = 1 << (L - 1).bit_length()

    # Forward FFTs (real-to-complex)
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Pointwise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get the linear convolution
    y = np.fft.irfft(Y, nfft)[:L]

    return y