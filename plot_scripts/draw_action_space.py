from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy import linspace

P = 2

# Generate points in the square action space
r_values = np.linspace(-1, 1, 10)
s_values = np.linspace(-1, 1, 10)

# Create a grid of points
r_grid, s_grid = np.meshgrid(r_values, s_values)

# Apply the transformation to each point in the grid

# Plot the original square action space
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Square Action Space')
plt.scatter(r_grid, s_grid, c='blue', alpha=0.5)
plt.xlabel('r')
plt.ylabel('s')
plt.axis('equal')

# Plot the transformed circle action space
plt.subplot(1, 2, 2)
plt.title(f'Transformed Action Space p={P}')


def norm(v: List[float], p=2):
    if sum([x ** p for x in v]) == 0:
        return 0
    return sum([x ** p for x in v]) ** (1 / p)


def transform(v: List[float], p=2):
    max_norm = max([abs(x) for x in v])
    factor = max_norm / norm(v, p=p)
    return [x * factor for x in v]


def invert(v: List[float], p=2):
    max_norm = max([abs(x) for x in v])
    factor = norm(v, p=p) / max_norm
    return [x * factor for x in v]


grid = linspace(-1, 1, 15)

# plotting
import matplotlib.pyplot as plt

for x in grid:
    for y in grid:
        w = transform([x, y], p=P)
        plt.arrow(x, y, w[0] - x, w[1] - y)
        plt.scatter([x], [y], color="blue", s=3)
        plt.scatter([w[0]], [w[1]], color="red", s=3)
plt.xlabel('r')
plt.ylabel('s')
plt.axis('equal')
plt.legend()

plt.tight_layout()
plt.show()
