import numpy as np
import matplotlib.pyplot as plt

# Beta values
betas = [0.5, 1.0, 2.0]
initial_eat_reward = 10

# Path length range
path_lengths = np.linspace(1, 400, 400)

# Create plots
plt.figure(figsize=(15, 5))

for i, beta in enumerate(betas):
    # Calculate rewards
    rewards_exp = initial_eat_reward * np.exp(-path_lengths / beta)
    rewards_pow = initial_eat_reward / np.power(path_lengths, 1 / beta)

    # Plotting
    plt.subplot(1, 3, i + 1)
    plt.plot(path_lengths, rewards_exp, label=r'Physics: $e^{-L/\beta}$', color='blue')
    plt.plot(path_lengths, rewards_pow, label=r'Math: $\frac{1}{L^{1/\beta}}$', color='green', linestyle='--')
    plt.title(f'Beta = {beta}')
    plt.xlabel('Path Length')
    plt.ylabel('Eat Reward')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

