import os
import sys

import numpy as np
import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
import torch as th
import yaml
from matplotlib import pyplot as plt
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.utils import get_model_path
from stable_baselines3.common.utils import set_random_seed

import worm_world.worms  # noqa: F401 pylint: disable=unused-import
from utils.functions import *


def create_plots(exp_id, env_name, plot_save_path, algo='ppo_lstm', folder='logs/', n_timesteps=1000, n_envs=1, image_rows=3,
                 image_cols=3,
                 verbose=1, no_render=False, deterministic=False, device='auto', load_best=False, num_threads=-1,
                 load_checkpoint=None, load_last_checkpoint=False, stochastic=False, norm_reward=False,
                 seed=None, reward_log='', render_mode=None, kwargs=None, hint='', file_postfix='',
                 overwrite_food_object_number=None) -> None:
    if overwrite_food_object_number is not None:
        if kwargs is not None:
            kwargs['min_number_food_objects'] = overwrite_food_object_number
            kwargs['max_number_food_objects'] = overwrite_food_object_number
        else:
            kwargs = {'min_number_food_objects': overwrite_food_object_number,
                      'max_number_food_objects': overwrite_food_object_number}
    try:
        _, model_path, log_path = get_model_path(exp_id, folder, algo, env_name, load_best,
                                                 load_checkpoint, load_last_checkpoint)
    except (AssertionError, ValueError) as e:
        if "rl-trained-agents" not in folder:
            raise e
        else:
            print(
                "Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
        exit()

    print(f"Loading {model_path}")
    if seed is not None:
        set_random_seed(seed)

    if num_threads > 0:
        if verbose > 1:
            print(f"Setting torch.num_threads to {num_threads}")
        th.set_num_threads(num_threads)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=norm_reward, test_mode=True)

    args_path = os.path.join(log_path, env_name, "yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    env_kwargs = {}
    if env_kwargs is not None:
        env_kwargs.update(kwargs)

    log_dir = reward_log if reward_log != "" else None

    if render_mode is not None:
        env_kwargs.update(render_mode=render_mode)

    env = create_test_env(
        env_name,
        n_envs=n_envs,
        stats_path=maybe_stats_path,
        seed=seed,
        log_dir=log_dir,
        should_render=not no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=seed)

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version or custom_objects:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    if "HerReplayBuffer" in hyperparams.get("replay_buffer_class", ""):
        kwargs["env"] = env

    model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=device, **kwargs)
    obs = env.reset()

    # Deterministic by default except for atari games
    stochastic = stochastic and not deterministic
    deterministic = not stochastic

    episode_rewards, episode_lengths = [], []
    # For HER, monitor success rate
    successes = []
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)
    #  create grid of 0.1 mm
    traces = 0
    steps = 0
    images = []
    image = None
    total_reward = 0
    for i in range(image_rows * image_cols + 1):
        # generator = range(n_timesteps)
        if i > 0:
            print(total_reward)
            images.append(image)
        if i == image_rows * image_cols:
            break
        env.reset()
        traces += 1
        done = False
        time_steps = 0
        while not done:
            image = env.render("rgb_array")
            action, lstm_states = model.predict(
                obs,  # type: ignore[arg-type]
                state=lstm_states,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            obs, reward, done, infos = env.step(action)
            total_reward = infos[0]['total_reward']
            # speed = infos[0]['speed']
            # rotation = infos[0]['rotation']
            if not no_render:
                env.render("rgb_array")
            steps += 1
            time_steps += 1
            if time_steps >= n_timesteps:
                break
            episode_start = done
    fig, axes = plt.subplots(image_rows, image_cols, figsize=(15, 15))
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(img)
        ax.axis('off')
    # plt.title(f'Traces model no {exp_id}')
    fig.suptitle(f'Traces model no {exp_id}\n' + hint)
    plt.tight_layout()
    plt.savefig(f'{plot_save_path}/model_{exp_id}_plots{file_postfix}.png', bbox_inches='tight')
    print(f'traces: {traces}, steps: {steps}')
    if verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    env.close()


if __name__ == "__main__":
    # algorithm = 'ppo'
    algorithm = 'ppo_lstm'
    # model_numbers = list(range(225, 227))
    # model_numbers = [267, 268]
    model_numbers = [1]
    # model_numbers = [211]

    load_best = False
    time_steps = 1_200
    errors = []

    for model_number in model_numbers:
        try:
            kwargs, hint = extract_kwargs(get_model_info_file_name(algorithm), model_number)
            print(kwargs)
            # kwargs['decay'] = True
            # print(kwargs)
            create_plots(env_name="WormWorld-v0", algo=algorithm, folder=get_log_directory(), exp_id=model_number,
                         no_render=False, reward_log='output', render_mode='rgb_array', n_timesteps=time_steps,
                         # overwrite_food_object_number=4,
                         # file_postfix='_best_model',
                         kwargs=kwargs, image_rows=3, image_cols=3, hint=wrap_string(hint, 100),
                         load_best=load_best, plot_save_path=get_save_plot_directory(algorithm))
        except Exception as e:
            print(e)
            errors.append(model_number)
    print(f'Errors in model numbers: {errors}')
