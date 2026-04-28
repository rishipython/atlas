import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1-D float64 arrays using FFT.
    Equivalent to np.convolve(x, h, mode='full') but typically much faster for
    large inputs.
    """
    # Ensure inputs are 1-D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    # Handle empty inputs explicitly
    if x.size == 0 or h.size == 0:
        return np.empty(0, dtype=np.float64)

    # Compute the convolution length
    conv_len = x.size + h.size - 1

    # Compute FFTs using real FFT for efficiency
    fft_x = np.fft.rfft(x, conv_len)
    fft_h = np.fft.rfft(h, conv_len)

    # Element-wise multiplication in frequency domain
    fft_product = fft_x * fft_h

    # Inverse real FFT to obtain the convolution result
    y = np.fft.irfft(fft_product, conv_len)

    return y