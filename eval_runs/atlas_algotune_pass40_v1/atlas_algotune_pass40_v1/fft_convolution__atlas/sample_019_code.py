import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1‑D real float64 arrays
    using the FFT.  The result is numerically equivalent to
    `np.convolve(x, h, mode='full')`.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (N,).
    h : np.ndarray
        Input array of shape (M,).

    Returns
    -------
    y : np.ndarray
        Convolution result of shape (N + M - 1,).
    """
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    # Handle empty inputs
    if x.size == 0 or h.size == 0:
        return np.array([], dtype=np.float64)

    N, M = x.size, h.size
    out_len = N + M - 1

    # Choose FFT size: next power of two >= out_len for speed
    nfft = 1 << (out_len - 1).bit_length()

    # Compute forward FFTs using real FFT for efficiency
    X = np.fft.rfft(x, nfft)
    H = np.fft.rfft(h, nfft)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse real FFT and truncate to desired output length
    y = np.fft.irfft(Y, nfft)[:out_len]

    return y