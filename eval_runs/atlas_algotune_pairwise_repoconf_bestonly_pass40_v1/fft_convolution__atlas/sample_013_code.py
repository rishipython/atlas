import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1‑D float64 arrays
    using an FFT‑based algorithm for speed.
    """
    # Ensure inputs are 1‑D and of dtype float64
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    n = x.size
    m = h.size
    out_len = n + m - 1

    # Handle empty inputs gracefully
    if out_len <= 0:
        return np.array([], dtype=np.float64)

    # Compute next power of two for efficient FFT (optional but often faster)
    nfft = 1 << (out_len - 1).bit_length()

    # FFT of both signals, zero‑padded to nfft
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Pointwise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get convolution result
    y = np.fft.irfft(Y, nfft)

    # Truncate to the correct output length
    return y[:out_len].astype(np.float64)