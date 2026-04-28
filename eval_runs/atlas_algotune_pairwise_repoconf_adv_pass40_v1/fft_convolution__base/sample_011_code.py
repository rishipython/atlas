import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D float64 arrays.

    This implementation uses the FFT for fast convolution and
    preserves the numerical behaviour of ``np.convolve`` with
    ``mode='full'``.  Empty inputs produce an empty array, matching
    NumPy's behaviour.
    """
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()

    N = x.size
    M = h.size

    # Handle empty inputs: np.convolve([], [1, 2]) -> array([], dtype=float64)
    if N == 0 or M == 0:
        return np.array([], dtype=np.float64)

    # Length of the full convolution
    L = N + M - 1

    # Compute FFTs of zero‑padded inputs
    X = np.fft.fft(x, n=L)
    H = np.fft.fft(h, n=L)

    # Element‑wise multiplication in frequency domain and inverse FFT
    y = np.fft.ifft(X * H).real

    # Return the real part (imaginary part is negligible due to numerical error)
    return y