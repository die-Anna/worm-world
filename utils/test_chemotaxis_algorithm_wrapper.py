import matplotlib.pyplot as plt
import numpy as np
from chemotaxis_algorithm_wrapper import ChemotaxisAlgorithmWrapper


def create_gd_data(num_episodes, initial_step_size, momentum, no_render=True, n_timesteps=4000, test=1):
    env_wrapper = ChemotaxisAlgorithmWrapper(initial_step_size=initial_step_size, momentum=momentum)
    env = env_wrapper.create_env()
    episode_rewards, episode_lengths = [], []
    episode_reward = 0
    episode_length = 0
    for i in range(num_episodes + 1):
        if i > 0:
            print(f'Episode {i}: Reward: {episode_reward}, Length: {episode_length}')
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
            if test == 1:
                action = env_wrapper.get_action(obs)
            else:
                action = env_wrapper.get_action2(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            if not no_render:
                env.render("rgb_array")
            if time_steps >= n_timesteps:
                break
            time_steps += 1
    env.close()
    return episode_rewards, episode_lengths


find_params = True
results = {}
if find_params:
    # Parameters to test
    step_sizes = [0.05, 0.1, 0.15, 2.0]
    momenta = [0.8, 0.85, 0.9,]

    num_episodes = 500

    # Data collection
    if False:
        for step_size in step_sizes:
            for momentum in momenta:
                rewards, lengths = create_gd_data(num_episodes, step_size, momentum)
                results[(step_size, momentum)] = (np.mean(rewards), np.mean(lengths))
    else:
        rewards, lengths = create_gd_data(num_episodes, 0.15, 1, test=1)
        results[(0.15, 1)] = (np.mean(rewards), np.mean(lengths))
        rewards, lengths = create_gd_data(num_episodes, 0.15, 1, test=2 )
        results[(0.15, 2)] = (np.mean(rewards), np.mean(lengths))
else:
    num_episodes = 100
    step_size = 0.15
    momentum = 0.55
    for use_momentum in [True, False]:
        rewards, lengths = create_gd_data(num_episodes, step_size, momentum, use_momentum=use_momentum)
        results[(step_size, 1 if use_momentum else 0)] = (np.mean(rewards), np.mean(lengths))

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(12, 12))
for idx, ((step_size, momentum), (avg_reward, avg_length)) in enumerate(results.items()):
    label = f"Step Size {step_size}, Momentum {momentum}"
    axs[0].bar(idx, avg_reward, label=label)
    axs[1].bar(idx, avg_length, label=label)

axs[0].set_title("Average Reward by Parameters")
axs[0].set_ylabel("Average Reward")
axs[1].set_title("Average Episode Length by Parameters")
axs[1].set_ylabel("Average Episode Length")

axs[0].set_xticks(np.arange(len(results)))
axs[0].set_xticklabels([f"{s}, {m}" for s, m in results.keys()], rotation=45, ha='right')
axs[1].set_xticks(np.arange(len(results)))
axs[1].set_xticklabels([f"{s}, {m}" for s, m in results.keys()], rotation=45, ha='right')

# Improve legend placement
fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()

# Determining the best parameters
best_reward_params = max(results.items(), key=lambda x: x[1][0])
best_length_params = min(results.items(), key=lambda x: x[1][1])

print("Best Parameters for Highest Reward: Step Size = {}, Momentum = {}, Average Reward = {}".format(
    best_reward_params[0][0], best_reward_params[0][1], best_reward_params[1][0]))
print("Best Parameters for Shortest Episode Length: Step Size = {}, Momentum = {}, Average Length = {}".format(
    best_length_params[0][0], best_length_params[0][1], best_length_params[1][1]))
