import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of x and h using FFT."""
    # Ensure inputs are float64 numpy arrays
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    # Handle empty input cases
    if x.size == 0 or h.size == 0:
        return np.array([], dtype=np.float64)

    # Length of the full convolution
    n = x.size
    m = h.size
    conv_len = n + m - 1

    # Next power of two for efficient FFT (optional but speeds up)
    fft_len = 1 << (conv_len - 1).bit_length()

    # Perform real FFTs, multiply, then inverse real FFT
    X = np.fft.rfft(x, fft_len)
    H = np.fft.rfft(h, fft_len)
    Y = X * H
    y = np.fft.irfft(Y, fft_len)

    # Return only the required convolution length
    return y[:conv_len]