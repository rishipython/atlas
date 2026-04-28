import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D arrays using FFT.

    Parameters
    ----------
    x : np.ndarray
        First input array (length N).
    h : np.ndarray
        Second input array (length M).

    Returns
    -------
    np.ndarray
        Array of length N + M - 1 containing the linear convolution
        y[n] = sum_k x[k] * h[n - k].

    Notes
    -----
    The implementation uses the real‑valued FFT (rfft/irfft) for
    efficiency and returns a float64 array that is numerically close
    to ``np.convolve(x, h, mode='full')``.
    """
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.float64).reshape(-1)

    n = x.size + h.size - 1
    # Next power of two for efficient FFT (optional, but often faster)
    size = 1 << (n - 1).bit_length()

    # Compute FFTs
    X = np.fft.rfft(x, size)
    H = np.fft.rfft(h, size)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT and truncate to the true convolution length
    y = np.fft.irfft(Y, size)[:n]

    return y