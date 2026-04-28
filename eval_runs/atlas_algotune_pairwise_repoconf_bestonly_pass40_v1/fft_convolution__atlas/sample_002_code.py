import numpy as np

def conv1d(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Return the full linear convolution of two 1‑D float64 arrays using FFT."""
    # Ensure inputs are 1‑D float64 arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    h = np.asarray(h, dtype=np.float64).ravel()
    N, M = x.size, h.size
    out_len = N + M - 1

    # Fastest FFT size: next power of two >= out_len
    fft_len = 1 << (out_len - 1).bit_length()

    # Compute FFTs
    X = np.fft.fft(x, fft_len)
    H = np.fft.fft(h, fft_len)

    # Element‑wise multiplication in frequency domain
    Y = X * H

    # Inverse FFT to get convolution result
    y = np.fft.ifft(Y).real

    # Return only the needed portion and ensure dtype float64
    return y[:out_len].astype(np.float64)