import os
import sys

import imageio
import numpy as np
import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
import torch as th
import yaml
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.utils import get_model_path
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed
import worm_world.worms  # noqa: F401 pylint: disable=unused-import
from functions import *
import pandas as pd


def make_video(exp_id, env_name, csv_save_path, path='videos/', algo='ppo_lstm', folder='logs/',
               n_timesteps=1000, n_envs=1, overwrite_food_object_number=None,
               verbose=1, deterministic=False, device='auto', load_best=False, num_threads=-1,
               load_checkpoint=None, load_last_checkpoint=False, stochastic=False, norm_reward=False,
               seed=None, reward_log='', progress=True, kwargs=None) -> None:  # noqa: C901
    if overwrite_food_object_number is not None:
        if kwargs is not None:
            kwargs['min_number_food_objects'] = overwrite_food_object_number
            kwargs['max_number_food_objects'] = overwrite_food_object_number
        else:
            kwargs = {'min_number_food_objects': overwrite_food_object_number,
                      'max_number_food_objects': overwrite_food_object_number}

    observations = []
    rewards = []
    speed = []
    rotation = []
    agent_locations = []

    _, model_path, log_path = get_model_path(exp_id, folder, algo, env_name, load_best,
                                             load_checkpoint, load_last_checkpoint)
    if seed is not None:
        set_random_seed(seed)

    if num_threads > 0:
        if verbose > 1:
            print(f"Setting torch.num_threads to {num_threads}")
        th.set_num_threads(num_threads)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    if env_kwargs is not None:
        env_kwargs.update(kwargs)
    args_path = os.path.join(log_path, env_name, "yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if env_kwargs is not None:
        env_kwargs.update(env_kwargs)

    log_dir = reward_log if reward_log != "" else None

    env_kwargs.update(render_mode="rgb_array")
    env = create_test_env(
        env_name,
        n_envs=n_envs,
        stats_path=maybe_stats_path,
        seed=seed,
        log_dir=log_dir,
        should_render=True,
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

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)
    generator = range(n_timesteps)
    if progress:
        if tqdm is None:
            raise ImportError("Please install tqdm and rich to use the progress bar")
        generator = tqdm(generator)
    images = []
    try:
        for _ in generator:
            action, lstm_states = model.predict(
                obs,  # type: ignore[arg-type]
                state=lstm_states,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            obs, reward, done, info = env.step(action)
            observations.append(obs[0][-1])
            rewards.append(reward[0])
            speed.append(info[0]['speed'])
            rotation.append(info[0]['rotation'])
            agent_locations.append([info[0]['agent_location'][0], info[0]['agent_location'][1]])

            episode_start = done

            images.append(env.render("rgb_array"))
            episode_reward += reward[0]
            ep_len += 1

            if n_envs == 1:
                if done and verbose > 0:
                    # NOTE: for env using VecNormalize, the mean reward
                    # is a normalized reward when `--norm_reward` flag is passed
                    print(f"Episode Reward: {episode_reward:.2f}")
                    print("Episode Length", ep_len)
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(ep_len)
                    episode_reward = 0.0
                    ep_len = 0
        imageio.mimsave(path + f'movie_{exp_id}.gif', images)
    except KeyboardInterrupt:
        pass
    df_initial = pd.DataFrame({
        'speed': speed,
        'observation': observations,
        'rotation': np.array(rotation)
    })

    df_initial.to_csv(csv_save_path + f'/data_model_long_{model_number}.csv')
    if verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    env.close()


if __name__ == "__main__":
    algorithm = 'ppo_lstm'
    # algorithm = 'ppo'
    model_numbers = [1]
    load_best = False

    errors = []
    for model_number in model_numbers:
        # try:
        kwargs, hint = extract_kwargs(get_model_info_file_name(algorithm), model_number)
        kwargs['width'] = 60
        kwargs['height'] = 60
        kwargs['centroids'] = [[20, 20, 5., 5.], [40, 40, 5., 5.]]
        kwargs['start_position'] = [15, 40]
        kwargs['start_angle'] = 0.4
        make_video(env_name="WormWorld-v0", csv_save_path=get_csv_directory(algorithm),
                   path=get_save_video_directory(algorithm) + '/', algo=algorithm,
                   folder=get_log_directory(), exp_id=model_number, overwrite_food_object_number=2,
                   reward_log='output', n_timesteps=3000, kwargs=kwargs, load_best=load_best)

        # except Exception as e:
        #     print(e)
        #     errors.append(model_number)
    print(f'Errors in model numbers: {errors}')
