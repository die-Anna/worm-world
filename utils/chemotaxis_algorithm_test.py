import gymnasium
import numpy as np
from matplotlib import pyplot as plt
from functions import *
import worm_world.worms  # noqa: F401 pylint: disable=unused-import
from utils.chemotaxis_algorithm_wrapper import ChemotaxisAlgorithmWrapper


def chemotaxis_algorithm_test(iterations=2000):
    wrapper = ChemotaxisAlgorithmWrapper()
    env = wrapper.create_env()
    _ = env.reset()
    action_solution = np.random.uniform(-1, 1, size=(2,))
    # action_solution = np.array([0.9, 0.0]) # initialize wi
    velocity = np.zeros_like(action_solution)  # to store the momentum
    # action_solution[0] = 0.9  # set speed to 0.9
    # action_solution[1] = 0
    episode_reward = 0
    for i in range(iterations):
        obs, reward, terminated, truncated, info = env.step(action_solution)
        episode_reward += reward
        gradient = obs
        action_solution = wrapper.get_action(gradient)
        # Ensure actions are within bounds
        action_solution = np.clip(action_solution, -1, 1)

        env.render()
        if terminated or truncated or i == iterations - 2:
            img = env.render()
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{get_save_plot_directory("chemotaxis")}/model_chemotaxis_plots.png', bbox_inches='tight')
            break

    return episode_reward


num_steps = 2000  # Define number of steps to run

# Run learning over defined number of steps
total_reward = chemotaxis_algorithm_test(iterations=num_steps)
print(f"Total Reward after {num_steps} steps: {total_reward}")

