import numpy as np


def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of 1‑D real arrays `x` and `h`
    using the FFT.  The result is numerically equivalent to
    `np.convolve(x, h, mode='full')` for float64 inputs.
    """
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64, order="C")
    h = np.asarray(h, dtype=np.float64, order="C")

    N, M = x.size, h.size
    # Length of the output
    out_len = N + M - 1

    if out_len == 0:
        return np.array([], dtype=np.float64)

    # Next power of two for efficient FFT (optional but usually faster)
    L = 1 << (out_len - 1).bit_length()

    # Compute FFTs of both signals padded to length L
    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get convolution result
    y = np.fft.irfft(Y, n=L)

    # Truncate to the exact convolution length
    return y[:out_len]