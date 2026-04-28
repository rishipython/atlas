import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D float64 arrays using FFT.

    This implementation uses the FFT to compute the convolution in
    O((N+M) log(N+M)) time, which is much faster than the naive double
    loop for large inputs.

    Parameters
    ----------
    x : np.ndarray
        First input array of shape (N,).
    h : np.ndarray
        Second input array of shape (M,).

    Returns
    -------
    y : np.ndarray
        Full linear convolution of shape (N + M - 1,).
    """
    # Ensure input is 1‑D and of float64 dtype
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N, M = x.size, h.size

    # Handle trivial cases
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of the linear convolution
    conv_len = N + M - 1

    # Choose FFT length as next power of two for speed
    fft_len = 1 << (conv_len - 1).bit_length()

    # Compute FFTs
    X = np.fft.fft(x, fft_len)
    H = np.fft.fft(h, fft_len)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to obtain convolution result
    y = np.fft.ifft(Y)

    # Take real part (imaginary part should be negligible)
    y = np.real(y)

    # Truncate to the exact convolution length
    return y[:conv_len]