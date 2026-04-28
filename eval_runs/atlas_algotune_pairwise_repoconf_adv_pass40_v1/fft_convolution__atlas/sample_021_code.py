import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return ``np.convolve(x, h, mode='full')`` but MUCH faster using FFT."""
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N = x.shape[0]
    M = h.shape[0]
    out_len = N + M - 1

    # Handle empty inputs: numpy.convolve returns an empty array
    if out_len <= 0:
        return np.array([], dtype=np.float64)

    # Choose an FFT length that is a fast size (power‑of‑two or similar)
    nfft = np.fft.next_fast_len(out_len)

    # Perform convolution via real FFTs
    fft_x = np.fft.rfft(x, n=nfft)
    fft_h = np.fft.rfft(h, n=nfft)
    fft_y = fft_x * fft_h

    y = np.fft.irfft(fft_y, n=nfft)[:out_len]
    return y