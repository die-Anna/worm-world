import numpy as np
import matplotlib.pyplot as plt

L = np.linspace(0.1, 10, 400)  # Avoid zero to prevent division errors
beta = 2.0  # Example beta value

physics_func = np.exp(-L / beta)
math_func = 1 / np.power(L, 1 / beta)

fig, ax = plt.subplots(figsize=(8, 6))

# Plotting both functions on the same graph
ax.plot(L, physics_func, label=r'Physics: $e^{-L/\beta}$', color='blue')
ax.plot(L, math_func, label=r'Math: $\frac{1}{L^{1/\beta}}$', color='green')
ax.set_title("Reward Type", fontsize=20)
ax.set_xlabel('Path length', fontsize=18)
ax.set_ylabel('Reward', fontsize=18)
ax.legend(fontsize=18)

plt.tight_layout()
plt.show()
