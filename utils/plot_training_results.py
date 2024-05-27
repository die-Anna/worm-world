from math import ceil
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results
import pandas as pd

from functions import *

X_TIMESTEPS = "timesteps"
X_EPISODES = "episodes"
X_WALLTIME = "walltime_hrs"
X_AVERAGE = "average_values"
X_RUNNING_AVERAGE = "running_average"

POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]

def get_axis_data(data_frame: pd.DataFrame, x_axis: str, block_size = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a data frame variable to x ans ys

    :param data_frame: the input data
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param block_size: used to define the size of averaging blocks
    :return: the x and y output
    """
    if x_axis == X_TIMESTEPS:
        print(X_TIMESTEPS)
        x_var = np.cumsum(data_frame.l.values)
        y_var = data_frame.r.values
    elif x_axis == X_EPISODES:
        print(X_EPISODES)
        print(data_frame)
        print(len(data_frame))
        print(f'cumsum: {np.cumsum(data_frame.l.values)}')
        x_var = np.arange(len(data_frame))
        y_var = data_frame.r.values
    elif x_axis == X_WALLTIME:
        print(X_WALLTIME)
        # Convert to hours
        x_var = data_frame.t.values / 3600.0
        y_var = data_frame.r.values
    elif x_axis == X_AVERAGE:
        arr_x = data_frame.l.values
        rest_x = len(arr_x) % block_size
        arr_x = arr_x[rest_x:]
        print(len(arr_x))
        print(f'rest x (len: {len(arr_x)}): {rest_x}')
        x_var = np.mean(arr_x[:(len(arr_x) // block_size) * block_size].reshape(-1, block_size), axis=1)
        arr_y = data_frame.r.values
        rest_y = len(arr_y) % block_size
        arr_y = arr_y[rest_y:]
        print(f'rest y (len: {len(arr_y)}): {rest_y}')
        y_var = arr_y[::block_size]
    elif x_axis == X_RUNNING_AVERAGE:
        print(data_frame.l.values.shape, data_frame.r.values.shape)
        x_var = np.cumsum(data_frame.l.values)
        y_var = np.convolve(data_frame.r.values, np.ones(block_size)/block_size, mode='same')
    else:
        raise NotImplementedError
    print(f'x_var: {x_var.shape}')
    print(f'y_var: {y_var.shape}')
    return x_var, y_var


def evaluate_results(log_dir, model_no, hint, plot_save_path, x_axis=X_RUNNING_AVERAGE, block_size=20):
    # Load monitor logs
    results = load_results(log_dir)
    print(results)
    # Use 'timesteps' or 'episodes' as the x-axis variable
    timesteps, mean_rewards = get_axis_data(results, x_axis, block_size)
    cutting_length = 0
    # Check if 'l' (episode lengths) is available in the monitor logs
    if 'l' in results:
        episode_lengths = results['l']
        if x_axis == X_AVERAGE:
            episode_lengths = results['t']
            print(episode_lengths)
            rest = len(episode_lengths) % block_size
            episode_lengths = episode_lengths[rest::block_size]
        if x_axis == X_RUNNING_AVERAGE:
            episode_lengths = np.convolve(results['l'], np.ones(block_size)/block_size, mode='valid')
            cutting_length = ceil((len(results['l']) - len(episode_lengths)) / 2.0)
            episode_lengths = np.convolve(results['l'], np.ones(block_size)/block_size, mode='same')

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        if x_axis == X_AVERAGE:
            plt.plot(episode_lengths, mean_rewards, label=f'Mean Reward blocks({block_size})')
        elif x_axis == X_RUNNING_AVERAGE:
            plt.plot(timesteps[cutting_length:-cutting_length],
                     mean_rewards[cutting_length:-cutting_length],
                     label=f'Running Mean Reward ({block_size} episodes)')
        else:
            plt.plot(timesteps, mean_rewards, label='Mean Reward')
        plt.title(f'Evaluation of Training Results (model {model_no})\n' + hint)
        plt.xlabel(f'{x_axis.capitalize()}')
        plt.ylabel('Rewards')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        if x_axis == X_AVERAGE:
            plt.plot(episode_lengths, timesteps, label=f'Episode Length blocks({block_size})', color='orange')
        elif x_axis == X_RUNNING_AVERAGE:
            plt.plot(timesteps[cutting_length:-cutting_length],
                     episode_lengths[cutting_length:-cutting_length],
                     label=f'Running Episode Length ({block_size} episodes)', color='orange')
        else:
            plt.plot(timesteps, episode_lengths, label='Episode Length', color='orange')
        plt.xlabel(f'{x_axis.capitalize()}')
        plt.ylabel('Episode Length')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'{plot_save_path}/model_{model_no}_results.png', bbox_inches='tight')

        # Print final statistics
        print(f"Total {x_axis.capitalize()}: {timesteps[-1]}")
        print(f"Mean Reward: {np.mean(mean_rewards)}")
        print(f"Best Mean Reward: {np.max(mean_rewards)}")
        print(f"Mean Episode Length: {np.mean(episode_lengths)}")
        print(f"Max Episode Length: {np.max(episode_lengths)}")
    else:
        # Plot only mean rewards if 'l' (episode lengths) is not available
        plt.figure(figsize=(12, 6))
        plt.plot(timesteps, mean_rewards, label='Mean Reward')
        plt.title(f'Evaluation of Training Results (model {model_no})\n' + hint)
        plt.xlabel(f'{x_axis.capitalize()}')
        plt.ylabel('Rewards')
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(f'{plot_save_path}/model_{model_no}_results.png', bbox_inches='tight')

        # Print final statistics
        print(f"Total {x_axis.capitalize()}: {timesteps[-1]}")
        print(f"Mean Reward: {np.mean(mean_rewards)}")
        print(f"Best Mean Reward: {np.max(mean_rewards)}")


# Example usage
if __name__ == "__main__":
    # algorithm = 'ppo'
    algorithm = 'ppo_lstm'
    # model_numbers = list(range(225, 228))
    # model_numbers = [262]
    model_numbers = [1]

    # model_numbers = [204]

    block_size = 50
    errors = []

    for model_number in model_numbers:
        try:
            print(f'model_number: {model_number}')
            kwargs, hint = extract_kwargs(get_model_info_file_name(algorithm), model_number)
            evaluate_results(get_model_directory(algorithm, model_number), model_no=model_number,
                             hint=wrap_string(hint, 100), block_size=block_size,
                             plot_save_path=get_save_plot_directory(algorithm))
        except Exception as e:
            print(f'error: {e}')
            errors.append(model_number)
    print(f'Errors in model numbers: {errors}')
