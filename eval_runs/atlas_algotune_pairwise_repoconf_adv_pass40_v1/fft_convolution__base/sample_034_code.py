import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of x and h using FFT.

    This implementation is equivalent to `np.convolve(x, h, mode='full')`
    but is much faster for large inputs.  The result is numerically close
    to the reference implementation within typical floating‑point tolerances.
    """
    # Ensure input is 1‑D and of type float64
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.size, h.size
    out_len = N + M - 1

    # Choose FFT length as the next power of two >= out_len
    fft_len = 1 << (out_len - 1).bit_length()

    # Compute FFTs
    X = np.fft.fft(x, fft_len)
    H = np.fft.fft(h, fft_len)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get convolution result
    y = np.fft.ifft(Y).real

    # Return only the first out_len samples
    return y[:out_len]