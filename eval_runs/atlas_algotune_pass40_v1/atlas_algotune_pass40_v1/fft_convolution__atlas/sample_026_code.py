import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return the full linear convolution of two 1‑D float64 arrays using an FFT‑based method.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (N,).
    h : np.ndarray
        Input array of shape (M,).

    Returns
    -------
    y : np.ndarray
        The full linear convolution of ``x`` and ``h`` of shape (N + M - 1,).
    """
    # Ensure input is a NumPy array with float64 dtype
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N = x.shape[0]
    M = h.shape[0]

    # Handle empty inputs gracefully
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of the linear convolution result
    out_len = N + M - 1

    # Choose FFT length: next power of two for speed
    fft_len = 1 << (out_len - 1).bit_length()

    # Compute FFTs of the padded inputs
    X = np.fft.rfft(x, n=fft_len)
    H = np.fft.rfft(h, n=fft_len)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get convolution result
    y_full = np.fft.irfft(Y, n=fft_len)

    # Truncate to the exact linear convolution length
    return y_full[:out_len]