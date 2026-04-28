# EVOLVE-BLOCK-START
import numpy as np
import math


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Hexagonal packing: 4 rows with 6,5,5,5 circles respectively
    # Compute radius from perimeter constraint w + h = 2
    sqrt3 = math.sqrt(3.0)
    # Determine the number of rows and maximum columns
    # 5‑row hexagonal layout gives a slightly larger radius for 21 circles
    rows = [5, 4, 5, 4, 3]
    max_cols = max(rows)
    nrows = len(rows)
    # Width = 2*r + (max_cols-1)*2*r = 2*r*max_cols
    # Height = 2*r + (nrows-1)*sqrt(3)*r
    # Perimeter constraint: 2*(width+height)=4  →  width+height=2
    # Solve for r: r = 2 / (2*(max_cols+1) + (nrows-1)*sqrt3)
    denom = 2 * (max_cols + 1) + (nrows - 1) * sqrt3
    r = 2.0 / denom

    # Preallocate result array
    circles = np.empty((21, 3), dtype=float)

    # Row configuration: list of number of circles per row
    # Use the same 5‑row layout for placement as for the radius calculation
    rows = [5, 4, 5, 4, 3]
    # Horizontal offset for odd rows (hexagonal shift)
    # In a hexagonal packing the horizontal offset for odd rows is exactly one radius
    shift = r

    idx = 0
    for i, n_circles in enumerate(rows):
        # y-coordinate of the center of the row
        y = r + i * sqrt3 * r
        # Horizontal starting x
        if i % 2 == 0:  # even row, no shift
            x_start = r
        else:  # odd row, shifted
            x_start = r + shift
        for j in range(n_circles):
            x = x_start + j * 2 * r
            circles[idx] = [x, y, r]
            idx += 1

    # Sanity check: we should have placed exactly 21 circles
    if idx != 21:
        raise ValueError(f"Generated {idx} circles instead of 21")
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
