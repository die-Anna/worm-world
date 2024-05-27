import numpy as np
import matplotlib.pyplot as plt
import math

# Define parameters
step_estimate = 10000
steps = np.arange(1, 50001)  # Example range of steps from 1 to 50000
test_step_food_found = 25000  # Assume food was found at step 25000

# Calculate the reward penalty
penalty = np.exp((steps - test_step_food_found) / step_estimate)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(steps, penalty, label='Hunger Penalty')
plt.title('Hunger Penalty Over Time', fontsize=20)
plt.xlabel('Steps', fontsize=18)
plt.ylabel('Penalty', fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.show()
