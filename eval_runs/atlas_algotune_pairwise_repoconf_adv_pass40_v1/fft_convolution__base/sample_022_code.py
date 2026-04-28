import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D arrays using FFT.

    Parameters
    ----------
    x : np.ndarray
        First input array, shape (N,).
    h : np.ndarray
        Second input array, shape (M,).

    Returns
    -------
    y : np.ndarray
        Full convolution, shape (N + M - 1,).
    """
    # Ensure input is 1-D float64
    x = np.asarray(x, dtype=np.float64, order='C')
    h = np.asarray(h, dtype=np.float64, order='C')

    N, M = x.shape[0], h.shape[0]
    out_len = N + M - 1

    # Choose FFT length as next power of two for speed
    nfft = 1 << (out_len - 1).bit_length()

    # Compute FFTs
    X = np.fft.fft(x, nfft)
    H = np.fft.fft(h, nfft)

    # Element‑wise multiplication and inverse FFT
    Y = np.fft.ifft(X * H)

    # Return real part (imaginary part is numerically negligible)
    return np.real(Y[:out_len])