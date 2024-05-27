import numpy as np
import statsmodels
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from rl_zoo3.utils import get_model_path, get_saved_hyperparams, create_test_env, ALGOS
from stable_baselines3.common.utils import set_random_seed

import worm_world.worms  # noqa: F401 pylint: disable=unused-import
from functions import *
import pandas as pd
import statsmodels.api as sm
import seaborn as sns


def collect_and_visualize(exp_id, env_name, csv_save_path, algo='ppo', folder='logs/', load_best=False,
                          load_checkpoint=None, load_last_checkpoint=False, seed=None,
                          kwargs=None, render_mode=None, reward_log='', norm_reward=False,
                          no_render=True, device='auto', n_samples=10, label_hint='', plot_save_path=None):
    observations = []
    rewards = []
    speed = []
    rotation = []
    agent_locations = []

    _, model_path, log_path = get_model_path(exp_id, folder, algo, env_name, load_best,
                                             load_checkpoint, load_last_checkpoint)
    if seed is not None:
        set_random_seed(seed)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=norm_reward, test_mode=True)

    env_kwargs = {}
    if kwargs is not None:
        env_kwargs.update(kwargs)
    log_dir = reward_log if reward_log != "" else None
    if render_mode is not None:
        env_kwargs.update(render_mode=render_mode)
    env = create_test_env(
        env_name,
        n_envs=1,
        stats_path=maybe_stats_path,
        seed=seed,
        log_dir=log_dir,
        should_render=not no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )
    model = ALGOS[algo].load(model_path, device=device)

    observation = env.reset()

    # Collect activations, observations and movement
    for _ in range(n_samples):
        action, _ = model.predict(observation)
        observation, reward, done, info = env.step(action)
        rewards.append(reward[0])
        speed.append(info[0]['speed'])
        rotation.append(info[0]['rotation'])
        agent_locations.append([info[0]['agent_location'][0], info[0]['agent_location'][1]])
        if 'frame_stacking' in kwargs and kwargs['frame_stacking']:
            observations.append(observation[0][-1])
        else:
            observations.append(observation[0][0])

    df = pd.DataFrame({
        'speed': speed,
        'observation': observations,
        'rotation': np.array(rotation)
    })
    plot_heatmap(df, x_axis='speed', save_path=None)
    plot_heatmap(df, x_axis='rotation', save_path=None)


def plot_heatmap(df: pd.DataFrame, x_axis: str, num_bins=50, save_path: str = None):
    # df.head()
    # eliminate outliers
    df = df[(df['observation'] > -0.1) & (df['observation'] < 0.1)]

    # Calculate the 2D histogram
    print(df)
    h2d, x_edges, y_edges = np.histogram2d(df[x_axis], df['observation'], bins=num_bins)
    h2d = h2d.T

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(h2d, cmap='jet', cbar=True)

    # label every nth bin to prevent overcrowding
    n = 5
    rotation_ticks = np.arange(num_bins // n) * n
    rotation_tick_labels = ['{:.2f}'.format(edge) for edge in x_edges[rotation_ticks]]
    observation_ticks = np.arange(num_bins // n) * n
    observation_tick_labels = ['{:.2f}'.format(edge) for edge in y_edges[observation_ticks]]

    ax.set_xticks(rotation_ticks)
    ax.set_xticklabels(rotation_tick_labels, rotation=45)
    ax.set_yticks(observation_ticks)
    ax.set_yticklabels(observation_tick_labels)
    ax.set_xlabel(x_axis)
    ax.set_ylabel('Observation')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()  # Close the plot to prevent it from displaying in this output cell


if __name__ == '__main__':
    # algorithm = 'ppo'
    algorithm = 'ppo_lstm'
    # model_numbers = list(range(195, 198))
    model_numbers = [217]

    # model_numbers = [161]
    samples = 2000

    errors = []
    for model_number in model_numbers:
        # try:
        print(f'model_number: {model_number}')
        kw_args, hint = extract_kwargs(get_model_info_file_name(algorithm), model_number)
        collect_and_visualize(algo=algorithm, exp_id=model_number, env_name="WormWorld-v0", n_samples=samples,
                              kwargs=kw_args,
                              label_hint=wrap_string(hint, 100), folder=get_log_directory(),
                              plot_save_path=get_save_plot_directory(algorithm),
                              csv_save_path=get_csv_directory(algorithm))
        # except Exception as e:
        #     print(f'error: {e}')
        #     errors.append(model_number)
    print(f'Errors in model numbers: {errors}')
