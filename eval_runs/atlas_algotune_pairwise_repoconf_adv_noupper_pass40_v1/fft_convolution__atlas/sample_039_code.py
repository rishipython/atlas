import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D float64 arrays.

    This implementation uses the FFT to compute the convolution
    efficiently for large inputs.  It is numerically equivalent
    to ``np.convolve(x, h, mode='full')`` for all typical use
    cases.
    """
    # Ensure inputs are numpy arrays of type float64
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    N = x.shape[0]
    M = h.shape[0]
    out_len = N + M - 1

    # Handle degenerate cases
    if out_len <= 0:
        return np.empty(0, dtype=np.float64)

    # Compute next power of two for efficient FFT
    L = 1 << (out_len - 1).bit_length()

    # Forward FFTs
    X = np.fft.fft(x, L)
    H = np.fft.fft(h, L)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT and take real part
    y = np.fft.ifft(Y).real

    # Return the full linear convolution result
    return y[:out_len]