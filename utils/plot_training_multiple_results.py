from math import ceil

import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results

from functions import *
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, sem, t
import seaborn as sns


X_TIMESTEPS = "timesteps"
X_EPISODES = "episodes"
X_WALLTIME = "walltime_hrs"
X_AVERAGE = "average_values"
X_RUNNING_AVERAGE = "running_average"

POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]


def evaluate_results(log_dir, model_no, hint, plot_save_path, block_size=20):
    # Load monitor logs
    results = load_results(log_dir)
    timesteps = np.cumsum(results.l.values)
    mean_rewards = np.convolve(results.r.values, np.ones(block_size) / block_size, mode='same')

    episode_lengths = np.convolve(results['l'], np.ones(block_size) / block_size, mode='valid')
    cutting_length = ceil((len(results['l']) - len(episode_lengths)) / 2.0)
    episode_lengths = np.convolve(results['l'], np.ones(block_size) / block_size, mode='same')
    return timesteps, cutting_length, mean_rewards, episode_lengths, results.r.values


def plot_results(timesteps_array, cutting_length_array, mean_reward_array, episode_length_array,
                 plot_save_path, model_numbers, colors):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    for i in range(len(timesteps_array)):
        for timesteps, cutting_length, mean_rewards, model in zip(timesteps_array[i], cutting_length_array[i],
                                                                  mean_reward_array[i], model_numbers[i]):
            plt.plot(timesteps[cutting_length:-cutting_length],
                     mean_rewards[cutting_length:-cutting_length],
                     label=f'Model {model}', color=colors[i], linewidth=0.5)
    plt.title(f'Evaluation of Training Results')
    plt.xlabel(f'Running Average')
    plt.ylabel('Rewards')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for i in range(len(timesteps_array)):
        for timesteps, cutting_length, episode_lengths, model in zip(timesteps_array[i], cutting_length_array[i],
                                                                     episode_length_array[i], model_numbers[i]):
            plt.plot(timesteps[cutting_length:-cutting_length],
                     episode_lengths[cutting_length:-cutting_length],
                     label=f'Model {model} ', color=colors[i], linewidth=0.5)
    plt.xlabel(f'Running Average')
    plt.ylabel('Episode Length')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{plot_save_path}/model_results.png', bbox_inches='tight')


def perform_statistical_tests(mean_reward_array, model_numbers):
    print([len(rewards) for rewards in mean_reward_array[0]])
    print([len(rewards) for rewards in mean_reward_array[1]])
    start_evaluation = 1000
    first_group_rewards = np.concatenate([rewards[start_evaluation:]
                                          if len(rewards) > start_evaluation
                                          else rewards for rewards in mean_reward_array[0]])
    second_group_rewards = np.concatenate([rewards[start_evaluation:]
                                           if len(rewards) > start_evaluation
                                           else rewards for rewards in mean_reward_array[1]])
    # Perform T-test
    t_stat, p_value_ttest = ttest_ind(first_group_rewards, second_group_rewards, equal_var=False)

    # Mann-Whitney U Test for non-parametric data
    u_stat, p_value_mwu = mannwhitneyu(first_group_rewards, second_group_rewards, alternative='two-sided')

    # Confidence Intervals
    confidence = 0.95
    n1, n2 = len(first_group_rewards), len(second_group_rewards)
    stderr1, stderr2 = sem(first_group_rewards), sem(second_group_rewards)
    t_critical = t.ppf((1 + confidence) / 2., n1 + n2 - 2)
    margin_of_error1 = stderr1 * t_critical
    margin_of_error2 = stderr2 * t_critical
    ci_first = (np.mean(first_group_rewards) - margin_of_error1, np.mean(first_group_rewards) + margin_of_error1)
    ci_second = (np.mean(second_group_rewards) - margin_of_error2, np.mean(second_group_rewards) + margin_of_error2)

    # Standard Deviation
    std_dev1 = np.std(first_group_rewards)
    std_dev2 = np.std(second_group_rewards)

    # Visualization of Density Plots
    plt.figure(figsize=(10, 6))
    sns.kdeplot(first_group_rewards, label=f"Group 1 (n={n1}, std={std_dev1:.2f})", bw_adjust=0.5, fill=True)
    sns.kdeplot(second_group_rewards, label=f"Group 2 (n={n2}, std={std_dev2:.2f})", bw_adjust=0.5, fill=True)
    plt.title('Density Plot of Rewards')
    plt.xlabel('Rewards')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    print(f"T-test results: T-statistic = {t_stat}, P-value = {p_value_ttest}")
    print(f"Mann-Whitney U test results: U-statistic = {u_stat}, P-value = {p_value_mwu}")
    print(f"Confidence Interval for Group 1: {ci_first}")
    print(f"Confidence Interval for Group 2: {ci_second}")
    print(f"Standard Deviation for Group 1: {std_dev1}")
    print(f"Standard Deviation for Group 2: {std_dev2}")

    if p_value_ttest < 0.05 or p_value_mwu < 0.05:
        print("Statistical tests suggest that the second group of models might perform differently.")
    else:
        print("No significant difference found between the groups based on the tests.")

def plot_rewards_with_outliers(data):
    plt.style.use('ggplot')

    # Check and prepare the data, skipping the first 500 entries in each group
    prepared_data = []
    for group in data:
        if isinstance(group, (list, np.ndarray)):
            try:
                # Convert each subgroup to a flat, homogeneous numpy array, skipping the first 500 entries
                prepared_data.append(np.concatenate([np.array(subgroup[500:]).flatten() for subgroup in group]))
            except Exception as e:
                print(f"Error processing data group: {e}")
                continue  # Skip groups that cause errors

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(prepared_data, notch=True, patch_artist=True, widths=0.4,
                    flierprops=dict(marker='o', color='red', markersize=5))

    colors = ['#FF9999', '#FFCC99']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_title('Comparison of Model Rewards with Outliers')
    ax.set_ylabel('Rewards')
    ax.set_xticklabels([f"Group 1 (n={len(prepared_data[0])})", f"Group 2 (n={len(prepared_data[1])})"] if len(
        prepared_data) > 1 else ["Group 1"])
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    algorithm = 'ppo'
    # algorithm = 'ppo_lstm'
    # model_numbers = list(range(225, 228))
    # model_numbers = [[262, 263, 264, 265, 266, 267, 272, 273], [261, 260, 259, 258, 268, 269, 270, 271]]
    eat_memory_false = [280, 281, 282, 283, 288, 289, 290, 289]
    eat_memory_true = [278, 279, 285, 284, 286, 287, 292, 293]
    eat_memory_true_improved = [298, 299, 300, 301, 302, 303, 304, 305]
    model_numbers = [eat_memory_false, eat_memory_true_improved]
    hints = ['eat_memory=False', 'eat_memory=True']
    colors = ['blue', 'red']

    # model_numbers = [204]

    block_size = 50
    errors = []
    timesteps_array = []
    cutting_length_array = []
    mean_reward_array = []
    episode_length_array = []
    evaluated_models = []
    clean_rewards_outer = []
    for models, hint in zip(model_numbers, hints):
        timesteps_inner_array = []
        cutting_length_inner_array = []
        mean_reward_inner_array = []
        episode_length_inner_array = []
        evaluated_models_inner_array = []
        clean_rewards_inner = []
        for model_number in models:
            try:
                print(f'model_number: {model_number}')
                kwargs, hint = extract_kwargs(get_model_info_file_name(algorithm), model_number)
                timesteps, cutting_length, mean_rewards, episode_lengths, clean_rewards = evaluate_results(
                    get_model_directory(algorithm, model_number), model_no=model_number,
                    hint=wrap_string(hint, 100), block_size=block_size,
                    plot_save_path=get_save_plot_directory(algorithm))
                timesteps_inner_array.append(timesteps)
                cutting_length_inner_array.append(cutting_length)
                mean_reward_inner_array.append(mean_rewards)
                episode_length_inner_array.append(episode_lengths)
                evaluated_models_inner_array.append(model_number)
                clean_rewards_inner.append(clean_rewards)
            except Exception as e:
                print(f'error: {e}')
                errors.append(model_number)
        timesteps_array.append(timesteps_inner_array)
        cutting_length_array.append(cutting_length_inner_array)
        mean_reward_array.append(mean_reward_inner_array)
        episode_length_array.append(episode_length_inner_array)
        evaluated_models.append(evaluated_models_inner_array)
        clean_rewards_outer.append(clean_rewards_inner)

    plot_results(timesteps_array, cutting_length_array, mean_reward_array, episode_length_array,
                 get_save_plot_directory(algorithm), evaluated_models, colors)
    print(f'Errors in model numbers: {errors}')

    # Example usage: assuming the script that loads and calculates mean rewards populates `mean_reward_array` appropriately.
    # mean_reward_array = [mean_reward_array[0], mean_reward_array[1]]
    perform_statistical_tests(clean_rewards_outer, model_numbers)
    plot_rewards_with_outliers(clean_rewards_outer)
