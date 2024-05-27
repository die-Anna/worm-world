import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from rl_zoo3.utils import get_model_path, get_saved_hyperparams, create_test_env, ALGOS
from stable_baselines3.common.utils import set_random_seed
import worm_world.worms  # noqa: F401 pylint: disable=unused-import
from functions import *


def collect_and_visualize(exp_id, env_name, algo='ppo', folder='logs/', load_best=False,
                          load_checkpoint=None, load_last_checkpoint=False, seed=None,
                          kwargs=None, render_mode=None, reward_log='', norm_reward=False,
                          no_render=True, device='auto', n_samples=10, lable_hint='', plot_save_path=''):
    activations = {}
    observations = []
    rewards = []
    speed = []
    rotation = []
    agent_locations = []

    def get_activation(activation_name):
        def hook(_model, _input, output):
            if activation_name not in activations:
                activations[activation_name] = []
            if isinstance(output, tuple):
                for tensor in output:
                    # Ensure it's actually a tensor (safety check)
                    if torch.is_tensor(tensor):
                        activations[activation_name].append(tensor.detach().cpu().numpy())
                        # print('--------------')
                        # print(tensor.detach().cpu().numpy())
                        # print('+++++++++')
                    else:
                        print("Found a non-tensor object in the tuple.")
            else:
                activations[activation_name].append(output.detach().cpu().numpy())

        return hook

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
    print(model_path)
    model = ALGOS[algo].load(model_path, device=device)

    # Register hooks for all layers
    for name, layer in model.policy.named_children():
        print(f'name, layer: {name}, {layer}')
        layer.register_forward_hook(get_activation(name))

    observation = env.reset()
    lstm_states = None
    # Collect activations, observations and movement
    for _ in range(n_samples):
        action, lstm_states = model.predict(observation, lstm_states)
        # print(lstm_states)
        # input()
        observation, reward, done, info = env.step(action)
        rewards.append(reward[0])
        speed.append(info[0]['speed'])
        rotation.append(info[0]['rotation'])
        agent_locations.append([info[0]['agent_location'][0], info[0]['agent_location'][1]])
        if 'frame_stacking' in kwargs and kwargs['frame_stacking']:
            observations.append(observation[0][-1])
        else:
            observations.append(observation.squeeze(1))
        model.predict(observation)
    if algo == 'ppo':
        activations_matrix = np.array(activations['policy_net']).reshape(n_samples, -1)
    else:
        activations_matrix = np.array(activations['lstm_actor']).reshape(n_samples, -1)

    fig = plt.figure(figsize=(15, 18))
    fig.suptitle('Neuron Activation, Observation & Movement\n' + lable_hint)
    # Create a gridspec with 2 rows and 2 columns,
    # with the second column for color bars (invisible for the top plot)
    gs = GridSpec(5, 3, width_ratios=[1, 0.05, 0.3], height_ratios=[.5, 1, 0.05, .5, .5], hspace=0.2)

    # Plotting Observations on the top
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(observations, 'o-')
    ax0.axhline(0, color='black', linewidth=.5)
    ax0.set_title('Observations')
    ax0.set_ylabel('Observation Value')
    # ax0.set_ylim(min(-1.0, np.array(observations).min()), max(1.0, np.array(observations).max()))
    ax0.set_ylim(-1.0, 1.0)

    # Invisible axis for the color bar next to the observations plot
    cax0 = fig.add_subplot(gs[0, 1:])
    cax0.axis('off')

    # Plotting Activations on the bottom
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    im = ax1.imshow(activations_matrix.T, aspect='auto', cmap='viridis')
    if algo == 'ppo':
        ax1.set_title('Activations of lstm_actor')
    else:
        ax1.set_title('Activations of policy_net')
    ax1.set_ylabel('Neuron')

    # Color bar for the activations heatmap
    cax1 = fig.add_subplot(gs[1, 1])
    cbar = plt.colorbar(im, cax=cax1)
    cbar.set_label('Activation')
    c2ax2 = fig.add_subplot(gs[1, 2])
    c2ax2.axis('off')

    x, y = zip(*agent_locations)

    cax2 = fig.add_subplot(gs[2, 0])
    # Create a color bar at the bottom indicating the colors
    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=0, vmax=len(x)))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax2, orientation='horizontal')
    cbar.set_label('Step', rotation=0, labelpad=5)
    cbar.set_ticks(np.linspace(0, len(x), num=5))  # Adjust the number of ticks if needed
    cbar.set_ticklabels(range(0, len(x), len(x) // 5))  # Adjust the tick labels if needed
    cax2.axis('off')

    ax2 = fig.add_subplot(gs[3, 0], sharex=ax0)
    # speed_colors_2 = np.where(np.array(speed) > 0, 'red', 'green')
    ax2.plot(speed, '-', color='red', label='Speed')
    # ax2.scatter(range(len(speed)), speed, color=speed_colors_2)
    ax2.axhline(0, color='black', linewidth=.5)
    ax2.set_title('Speed')
    ax2.set_ylim(min(-1.0, np.array(speed).min()), max(1.0, np.array(speed).max()))
    cax2 = fig.add_subplot(gs[3:, 1:])

    # Plot the trace with a continuous line and multiple colors
    # Use a colormap to generate colors along the spectrum
    colors = plt.cm.rainbow(np.linspace(0, 1, len(x)))

    for i in range(len(x) - 1):
        cax2.plot(x[i:i + 2], y[i:i + 2], color=colors[i], linewidth=2)

    for spine in cax2.spines.values():
        spine.set_visible(True)
        spine.set_color('black')  # Set the color of the frame to black
        spine.set_linewidth(2)
    cax2.set_title('Trace')
    cax2.grid(True)
    # Set the aspect of the plot to be equal
    cax2.set_aspect('equal')

    # Remove axis for aesthetics similar to the example image
    cax2.axis('off')

    ax3 = fig.add_subplot(gs[4, 0], sharex=ax0)
    # rotation_colors = np.where(np.array(rotation) > 0, 'blue', 'yellow')
    ax3.plot(rotation, '-', color='grey', label='Rotation')
    ax3.axhline(0, color='black', linewidth=.5)
    # ax3.scatter(range(len(rotation)), rotation, color=rotation_colors)
    ax3.set_title('Rotation')
    ax3.set_ylim(min(-1.0, np.array(rotation).min()), max(1.0, np.array(rotation).max()))
    ax3.set_xlabel('Time (Sample Number)')

    # cax3 = fig.add_subplot(gs[3, 1])
    # cax3.axis('off')

    # Hide the x-axis labels for the top plot to avoid repetition
    # plt.setp(ax0.get_xticklabels(), visible=False)

    marker_points = [i for i, value in enumerate(rewards) if value > 0]
    print(marker_points)
    for point in marker_points:
        # Add vertical line on the observation plot
        ax0.axvline(x=point, color='red', linestyle='--', lw=2)

        # For the heatmap, the x-coordinates are scaled to the width of the activations matrix
        # We calculate the proportion of the point in the range of total samples
        proportion = point / n_samples
        # We then apply this proportion to the width of the matrix to get the correct x-coordinate
        x_coord = proportion * activations_matrix.T.shape[1]

        # Add vertical line on the activations heatmap
        ax1.axvline(x=x_coord, color='red', linestyle='--', lw=2)

    # plt.tight_layout()
    # plt.show()
    if plot_save_path == '':
        plt.show()
    else:
        plt.savefig(f'{plot_save_path}/model_{model_number}_activations.png', bbox_inches='tight')


if __name__ == '__main__':
    # algorithm = 'ppo'
    algorithm = 'ppo_lstm'
    # model_numbers = list(range(195, 198))
    model_numbers = [262]
    # model_numbers = [161]
    samples = 500
    load_best = True
    errors = []
    for model_number in model_numbers:
        # try:
        print(f'model_number: {model_number}')
        kw_args, hint = extract_kwargs(get_model_info_file_name(algorithm), model_number)
        collect_and_visualize(algo=algorithm, exp_id=model_number, env_name="WormWorld-v0", n_samples=samples,
                              kwargs=kw_args,
                              lable_hint=wrap_string(hint, 100), folder=get_log_directory(),
                              plot_save_path=get_save_plot_directory(algorithm), load_best=load_best)
        # except Exception as e:
        #    print(f'error: {e}')
        #    errors.append(model_number)
    print(f'Errors in model numbers: {errors}')
