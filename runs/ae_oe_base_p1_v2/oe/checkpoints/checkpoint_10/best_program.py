# EVOLVE-BLOCK-START
import numpy as np


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Use a 5x5 grid (25 positions) and take the first 21
    grid_size = 5
    spacing = 1.0 / grid_size  # grid spacing
    radius = spacing / 2.0     # radius fits within grid cell
    positions = [(i * spacing + spacing / 2,
                  j * spacing + spacing / 2)
                 for i in range(grid_size) for j in range(grid_size)]
    # Take the first 21 positions
    selected = positions[:21]
    circles = np.zeros((21, 3))
    for idx, (x, y) in enumerate(selected):
        circles[idx] = [x, y, radius]

    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
