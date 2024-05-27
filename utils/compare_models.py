import sys

import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
import torch as th
import yaml
from matplotlib import pyplot as plt
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.utils import get_model_path
from stable_baselines3.common.utils import set_random_seed

import worm_world.worms  # noqa: F401 pylint: disable=unused-import
from utils.functions import *
from chemotaxis_algorithm_wrapper import *
import scipy.stats as stats


def create_model_data(exp_id, env_name, algo='ppo_lstm', folder='logs/', n_timesteps=1000, n_envs=1,
                      num_episodes=50,
                      verbose=1, no_render=False, deterministic=False, device='auto', load_best=False, num_threads=-1,
                      load_checkpoint=None, load_last_checkpoint=False, stochastic=False, norm_reward=False,
                      seed=None, reward_log='', render_mode=None, kwargs=None,
                      overwrite_food_object_number=None, width=None, height=None):
    if overwrite_food_object_number is not None:
        if kwargs is not None:
            kwargs['min_number_food_objects'] = overwrite_food_object_number[0]
            kwargs['max_number_food_objects'] = overwrite_food_object_number[1]
        else:
            kwargs = {'min_number_food_objects': overwrite_food_object_number[0],
                      'max_number_food_objects': overwrite_food_object_number[1]}
    if width is not None and height is not None:
        if kwargs is not None:
            kwargs['width'] = width
            kwargs['height'] = height
        else:
            kwargs = {'width': width, 'height': height}

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
    _ = env.reset()

    # Deterministic by default except for atari games
    stochastic = stochastic and not deterministic
    deterministic = not stochastic

    episode_rewards, episode_lengths = [], []
    # For HER, monitor success rate
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)
    #  create grid of 0.1 mm
    steps = 0
    episode_reward = 0
    episode_length = 0
    for i in range(num_episodes + 1):
        # generator = range(n_timesteps)
        if i > 0:
            print(f'episode_reward: {episode_reward[0]}, episode_length:  {episode_length}')
            episode_rewards.append(episode_reward[0])
            episode_lengths.append(episode_length)
        if i == num_episodes:
            break
        total_reward = 0
        episode_reward = 0
        episode_length = 0
        obs = env.reset()
        done = False
        time_steps = 0
        while not done:
            action, lstm_states = model.predict(
                obs,  # type: ignore[arg-type]
                state=lstm_states,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            obs, reward, done, infos = env.step(action)
            episode_reward += reward
            episode_length += 1
            if not no_render:
                env.render("rgb_array")
            steps += 1
            time_steps += 1
            if time_steps >= n_timesteps:
                break
            episode_start = done
    env.close()
    return episode_rewards, episode_lengths


def create_gd_data(num_episodes, no_render=True, n_timesteps=1000,
                   min_number_food_objects=2, max_number_food_objects=4, height=35, width=35):
    env_wrapper = ChemotaxisAlgorithmWrapper(min_number_food_objects=min_number_food_objects,
                                             max_number_food_objects=max_number_food_objects,
                                             width=width, height=height)
    env = env_wrapper.create_env()
    episode_rewards, episode_lengths = [], []
    steps = 0
    episode_reward = 0
    episode_length = 0
    for i in range(num_episodes + 1):
        if i > 0:
            print(f'episode_reward: {episode_reward}, episode_length:  {episode_length}')
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        if i == num_episodes:
            break
        episode_reward = 0
        episode_length = 0
        obs, _ = env.reset()
        done = False
        time_steps = 0
        while not done:
            action = env_wrapper.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            if not no_render:
                env.render("rgb_array")
            steps += 1
            time_steps += 1
            if time_steps >= n_timesteps:
                break
    env.close()
    return episode_rewards, episode_lengths


if __name__ == "__main__":
    # first model
    algorithm_first_model = 'ppo'
    # algorithm_first_model = 'ppo_lstm'
    model_number_first_model = 308
    label_first_model = f'PPO (Model {model_number_first_model})'

    # second model
    # algorithm_second_model = 'ppo'
    algorithm_second_model = 'ppo_lstm'
    model_number_second_model = 262
    use_gd_second_model = False
    label_second_model = f'LSTM (Model {model_number_second_model})'

    if use_gd_second_model:
        label_second_model = 'Chemotaxis Algorithm'

    # params definitions
    num_episodes = 150
    load_best = False
    time_steps = 4_000
    errors = []

    initial_step_size = 0.1
    momentum = 0.6

    min_food_objects = 2
    max_food_objects = 4
    width = 40
    height = 40

    kwargs, hint = extract_kwargs(get_model_info_file_name(algorithm_first_model), model_number_first_model)
    print(kwargs)
    rewards_model_first_model, lengths_model_first_model = create_model_data(env_name="WormWorld-v0",
                                                                             algo=algorithm_first_model,
                                                                             folder=get_log_directory(),
                                                                             exp_id=model_number_first_model,
                                                                             no_render=True, reward_log='output',
                                                                             render_mode='rgb_array',
                                                                             n_timesteps=time_steps,
                                                                             kwargs=kwargs, num_episodes=num_episodes,
                                                                             load_best=load_best,
                                                                             overwrite_food_object_number=
                                                                             [min_food_objects, max_food_objects],
                                                                             width=width, height=height
                                                                             )
    if use_gd_second_model:
        rewards_second_model, lengths_second_model = create_gd_data(num_episodes=num_episodes, no_render=True,
                                                                    n_timesteps=time_steps,
                                                                    min_number_food_objects=min_food_objects,
                                                                    max_number_food_objects=max_food_objects,
                                                                    width=width, height=height)
    else:
        rewards_second_model, lengths_second_model = create_model_data(env_name="WormWorld-v0",
                                                                       algo=algorithm_first_model,
                                                                       folder=get_log_directory(),
                                                                       exp_id=model_number_first_model,
                                                                       no_render=True, reward_log='output',
                                                                       render_mode='rgb_array',
                                                                       n_timesteps=time_steps,
                                                                       kwargs=kwargs,
                                                                       num_episodes=num_episodes,
                                                                       load_best=load_best,
                                                                       overwrite_food_object_number=
                                                                       [min_food_objects, max_food_objects],
                                                                       width=width, height=height
                                                                       )
    # Calculate Descriptive Statistics
    print(f"{label_first_model} Rewards - Mean:", np.mean(rewards_model_first_model), "Std Dev:",
          np.std(rewards_model_first_model))
    print(f"{label_second_model} Rewards - Mean:", np.mean(rewards_second_model), "Std Dev:", np.std(rewards_second_model))

    print(f"{label_first_model} Lengths - Mean:", np.mean(lengths_model_first_model), "Std Dev:",
          np.std(lengths_model_first_model))
    print(f"{label_second_model} Lengths - Mean:", np.mean(lengths_second_model), "Std Dev:", np.std(lengths_second_model))

    # Statistical Tests
    # Rewards comparison
    t_stat, p_value_rewards = stats.mannwhitneyu(rewards_model_first_model, rewards_second_model,
                                                 alternative='two-sided')
    print("Rewards Comparison p-value:", p_value_rewards)

    # Episode Lengths comparison
    t_stat, p_value_lengths = stats.mannwhitneyu(lengths_model_first_model, lengths_second_model,
                                                 alternative='two-sided')
    print("Episode Lengths Comparison p-value:", p_value_lengths)

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Rewards
    axs[0, 0].hist(rewards_model_first_model, color='blue', alpha=0.7, label=label_first_model)
    axs[0, 0].hist(rewards_second_model, color='red', alpha=0.7, label=label_second_model)
    axs[0, 0].set_title('Reward Distribution')
    axs[0, 0].legend()

    # Lengths
    axs[0, 1].hist(lengths_model_first_model, color='blue', alpha=0.7, label=label_first_model)
    axs[0, 1].hist(lengths_second_model, color='red', alpha=0.7, label=label_second_model)
    axs[0, 1].set_title('Episode Length Distribution')
    axs[0, 1].legend()

    # Boxplots for Rewards
    axs[1, 0].boxplot([rewards_model_first_model, rewards_second_model], labels=[label_first_model, label_second_model])
    axs[1, 0].set_title('Reward Boxplot')

    # Boxplots for Lengths
    axs[1, 1].boxplot([lengths_model_first_model, lengths_second_model], labels=[label_first_model, label_second_model])
    axs[1, 1].set_title('Episode Length Boxplot')

    plt.tight_layout()
    plt.show()
