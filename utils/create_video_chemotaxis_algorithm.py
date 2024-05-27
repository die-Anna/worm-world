import imageio
import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from stable_baselines3.common.callbacks import tqdm
import worm_world.worms  # noqa: F401 pylint: disable=unused-import
import pandas as pd
from utils.functions import *
from chemotaxis_algorithm_wrapper import *

action_solution = None
velocity = None

def get_action(gradient, initial_step_size, momentum):
    global action_solution, velocity
    if action_solution is None:
        action_solution = np.random.uniform(-1, 1, size=(2,))
        velocity = np.zeros_like(action_solution)  # to store the momentum
    print(gradient)
    if gradient > 0:
        # Positive gradient: Encourage maintaining direction
        action_solution[0] = np.clip(action_solution[0] + initial_step_size, -1, 1)  # Maintain/increase speed
        action_solution[1] *= 0.9  # Reduce rotation
    else:
        # Negative gradient: Encourage direction change
        action_solution[0] = np.clip(action_solution[0] - initial_step_size, -1, 1)  # Reduce speed
        action_solution[1] = np.random.uniform(-1, 1)  # Randomize rotation

    velocity = momentum * velocity + initial_step_size * gradient
    action_solution += velocity

    # Ensure actions are within bounds
    action_solution = np.clip(action_solution, -1, 1)
    return action_solution


def make_video(env, wrapper, csv_save_path, video_path='videos/', n_timesteps=1000,
               verbose=1, progress=True) -> None:  # noqa: C901
    observations = []
    rewards = []
    speed = []
    rotation = []
    agent_locations = []

    obs, _ = env.reset()

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    generator = range(n_timesteps)
    if progress:
        if tqdm is None:
            raise ImportError("Please install tqdm and rich to use the progress bar")
        generator = tqdm(generator)
    images = []
    try:
        for _ in generator:
            action = wrapper.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            speed.append(info['speed'])
            rotation.append(info['rotation'])
            agent_locations.append([info['agent_location'][0], info['agent_location'][1]])

            images.append(env.render())
            episode_reward += reward
            ep_len += 1

            if truncated or terminated and verbose > 0:
                # NOTE: for env using VecNormalize, the mean reward
                # is a normalized reward when `--norm_reward` flag is passed
                print(f"Episode Reward: {episode_reward:.2f}")
                print("Episode Length", ep_len)
                episode_rewards.append(episode_reward)
                episode_lengths.append(ep_len)
                episode_reward = 0.0
                ep_len = 0
        imageio.mimsave(video_path + f'\movie_gd.gif', images)
    except KeyboardInterrupt:
        pass
    df_initial = pd.DataFrame({
        'speed': speed,
        'observation': observations,
        'rotation': np.array(rotation)
    })

    df_initial.to_csv(csv_save_path + f'/data_model_long_gd.csv')
    if verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    env.close()


if __name__ == "__main__":
    width = 60
    height = 60
    centroids = [[20, 20, 5., 5.], [40, 40, 5., 5.]]
    start_position = [15, 35]
    start_angle = 0.4

    """  beta = 2.0
    sigma = 3.0
    min_number_food_objects = 2
    max_number_food_objects = 4
    reward_type = 'math'
    test_eat_fixed_amount = True
    hunger_penalty = True
    gradient_reward = True

    use_multiplicative_noise = True
    use_additive_noise = True
    frame_stacking = False

    initial_step_size = 0.1
    # initial_step_size = 0.1
    momentum = 0.8"""
    # momentum = 0.6
    wrapper = ChemotaxisAlgorithmWrapper(width=width, height=height, centroids=centroids,
                                         start_position=start_position, start_angle=start_angle)
    env = wrapper.create_env()

    make_video(env, wrapper=wrapper, csv_save_path=get_csv_directory('gd'), video_path=get_save_video_directory('gd'))

