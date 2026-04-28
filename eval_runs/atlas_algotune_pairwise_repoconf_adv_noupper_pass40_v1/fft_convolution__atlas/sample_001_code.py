import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D float64 arrays.

    This implementation uses the FFT to achieve a significant speed
    improvement over a naïve double loop.  The result is numerically
    equivalent to ``np.convolve(x, h, mode='full')``.
    """
    # Handle empty input arrays – numpy.convolve returns an empty array
    if x.size == 0 or h.size == 0:
        return np.array([], dtype=np.float64)

    N, M = x.shape[0], h.shape[0]
    out_len = N + M - 1

    # Choose an FFT length that is a power of two (for speed) and
    # at least as large as the required output length.
    nfft = 1 << (out_len - 1).bit_length()

    # Compute the real FFTs of the zero‑padded inputs.
    X = np.fft.rfft(x, n=nfft)
    H = np.fft.rfft(h, n=nfft)

    # Element‑wise multiplication in the frequency domain.
    Y = X * H

    # Inverse real FFT to obtain the convolution result.
    y = np.fft.irfft(Y, n=nfft)

    # Truncate to the exact output length.
    return y[:out_len]