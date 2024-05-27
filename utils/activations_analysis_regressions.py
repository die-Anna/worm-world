import sys

import numpy as np
import statsmodels
import yaml
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from rl_zoo3.utils import get_model_path, get_saved_hyperparams, create_test_env, ALGOS
from stable_baselines3.common.utils import set_random_seed

import worm_world.worms  # noqa: F401 pylint: disable=unused-import
from functions import *
import pandas as pd
import statsmodels.api as sm

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# i = 0

def collect_and_visualize(exp_id, env_name, csv_save_path, algo='ppo_lstm', folder='logs/', load_best=True,
                          load_checkpoint=None, load_last_checkpoint=False, seed=None, stochastic=False,
                          kwargs=None, render_mode='rgb_array', reward_log='', norm_reward=False, deterministic=False,
                          no_render=True, device='auto', n_samples=10, label_hint='', plot_save_path='',
                          search_food=False):
    activations = {}
    observations = []
    rewards = []
    speed = []
    rotation = []
    agent_locations = []

    def get_activation(activation_name):
        def hook(_model, _inputs, output):
            # global i
            # if activation_name not in activations:
            if activation_name not in activations:
                activations[activation_name] = []
            # Detach and move to CPU for numpy conversion
            # print(output[1])
            # print(output[1][0])
            # print(output[1][1])
            # print(output)
            # exit()
            activations[activation_name].append(output[1][1].detach().cpu().numpy()[0])
            # activations[activation_name].append(inputs[0].detach().cpu().numpy()[0])
            # print(f'activations {inputs[0].detach().cpu().numpy()[0]}')
            # if i > 2:
            #     exit()
            # i += 1
            # output_data = output[-1][-1]  # take last layer
            # activations[activation_name].append(output_data.detach().cpu().numpy()[0][0])

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
    args_path = os.path.join(log_path, env_name, "yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    if env_kwargs is not None:
        env_kwargs.update(env_kwargs)
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

    print(model.policy)
    for name, layer in model.policy.named_children():
        if algo == 'ppo_lstm':
            if name == 'lstm_actor':
                layer.register_forward_hook(get_activation(name))
        else:
            if name == 'action_net':
                layer.register_forward_hook(get_activation(name))

    stochastic = stochastic and not deterministic
    deterministic = not stochastic
    observation = env.reset()
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)
    lstm_activations = []
    # Collect activations, observations and movement
    for _ in range(n_samples):
        action, lstm_states = model.predict(
            observation,  # type: ignore[arg-type]
            state=lstm_states,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        # shape of lstm_states: 2, 2, 1, 32
        # (actor & critic), 2 layers, one obs, 32 size
        # take actor, 1, layer
        lstm_activations.append(lstm_states[0][0][0])
        # lstm_activations.append(lstm_states[0][0])
        observation, reward, done, info = env.step(action)
        # print(f'data: {observation, reward, done, info}')
        rewards.append(reward[0])
        speed.append(info[0]['speed'])
        rotation.append(info[0]['rotation'])
        agent_locations.append([info[0]['agent_location'][0], info[0]['agent_location'][1]])
        if 'frame_stacking' in kwargs and kwargs['frame_stacking']:
            observations.append(observation[0][-1])
        else:
            observations.append(observation[0][0])
        # model.predict(observation)
    marker_points = [i for i, value in enumerate(rewards) if value > 0]
    if search_food and len(marker_points) == 0:
        return []
    if algo == 'ppo_lstm':
        # activations_matrix = np.array(activations['lstm_actor'])
        activations_matrix = np.array(lstm_activations)
    else:
        activations_matrix = np.array(activations['action_net'])
    # print(activations_matrix.shape)
    # exit()
    df_initial = pd.DataFrame({
        'speed': speed,
        'observation': observations,
        'rotation': np.array(rotation)
    })
    # print(activations_matrix.shape)
    print(activations_matrix.shape[1])
    matrix_columns = [f'neuron_{i + 1}' for i in range(activations_matrix.shape[1])]

    df_matrix = pd.DataFrame(activations_matrix, columns=matrix_columns)

    # Concatenate the two DataFrames horizontally
    final_df = pd.concat([df_initial, df_matrix], axis=1)
    final_df.to_csv(csv_save_path + f'/data_model_{model_number}.csv')
    start_point = 40
    length = 100
    point_of_interest = marker_points[0]
    if point_of_interest < start_point:
        return []
    begin = point_of_interest - start_point

    new_marker_points = [x - begin for x in marker_points]
    marker_points = [x for x in new_marker_points if x < length]
    final_df_shortened = final_df.iloc[begin: begin + length, :]
    y_speed_all = final_df['speed']
    y_rotation_all = final_df['rotation']
    y_observation_all = final_df['observation']

    y_speed_short = final_df_shortened['speed']
    y_rotation_short = final_df_shortened['rotation']

    # drop all columns except neurons
    X_all = final_df.drop(['speed', 'observation', 'rotation'], axis=1)
    X_short = final_df_shortened.drop(['speed', 'observation', 'rotation'], axis=1)

    # Add a constant to the model (intercept)
    X_all = statsmodels.api.add_constant(X_all)
    X_short = statsmodels.api.add_constant(X_short)

    model_speed = statsmodels.api.OLS(y_speed_all, X_all).fit()
    with open(csv_save_path + f'/summary_model_{model_number}_speed.txt', 'w') as fh:
        fh.write(model_speed.summary().as_text())

    prediction_vars = [[y_rotation_all, 'Rotation', y_rotation_short],
                       [y_speed_all, 'Speed', y_speed_short]]

    for p in prediction_vars:

        best_model = None
        best_r_squared = 0
        best_model_x = None
        best_col = None
        results = []
        for col in X_all.columns:
            if col.startswith('neuron'):
                x = X_all[[col]]
                y = p[0]
                x = statsmodels.api.add_constant(x)  # adding a constant
                model = statsmodels.api.OLS(y, x).fit()
                if best_model is None or model.rsquared > best_r_squared:
                    best_model = model
                    best_r_squared = model.rsquared
                    best_model_x = x
                    best_col = col
                results.append((col, model.rsquared, model.aic))  # Collecting R-squared and AIC

        # Sort results by the best fitting criterion, here R-squared (higher is better)
        results_sorted_by_r_squared = sorted(results, key=lambda k: k[1], reverse=True)
        print(f'best single neuron: {results_sorted_by_r_squared[0]}')
        # print("Sorted by R-squared:")

        # for result in results_sorted_by_r_squared:
        #     print(f"Neuron: {result[0]}, R-squared: {result[1]:.4f}, AIC: {result[2]:.2f}")

        model_test = statsmodels.api.OLS(p[0], X_all).fit()
        with open(csv_save_path + f'/summary_model_{model_number}_{p[1].lower()}.txt', 'w') as fh:
            fh.write(model_test.summary().as_text())

        predictions = model_test.predict(X_short)
        print(best_model_x)
        print(X_short[best_col])
        predictions_single_neuron = best_model.predict(best_model_x.iloc[begin: begin + length, :])

        plt.figure(figsize=(12, 6))  # Set the figure size as needed
        # Plot actual values
        plt.plot(p[2], label=f'Actual {p[1]}', color='blue', marker='o')  # 'o' adds markers to the line
        # Plot predicted values
        plt.plot(predictions, label=f'Predicted {p[1]}', color='red', linestyle='--',
                 marker='x')
        # plot prediction of single neuron
        plt.plot(predictions_single_neuron, label='Prediction Single Neuron', color='orange',
                 linestyle='--', marker='x')
        # Adding titles and labels
        plt.title(f'Comparison of Actual and Predicted {p[1]} (Model {model_number})')
        plt.xlabel('Index')
        plt.ylabel(f'{p[1]} Value')
        plt.legend()  # This adds a legend to distinguish between actual and predicted

        # Adding grid for better readability
        plt.grid(True)

        # Show plot
        if plot_save_path == '':
            plt.show()
        else:
            plt.savefig(f'{plot_save_path}/model_{model_number}_comparison_to_prediction_{p[1]}.png',
                        bbox_inches='tight')

    model_observation = statsmodels.api.OLS(y_observation_all, X_all).fit()
    with open(csv_save_path + f'/summary_model_{model_number}_observation.txt', 'w') as fh:
        fh.write(model_observation.summary().as_text())

        print(model_speed.summary())

    fig = plt.figure(figsize=(15, 18))
    fig.suptitle(f'Neuron Activation, Observation & Movement (Model {model_number})\n' + label_hint)
    # Create a gridspec with 2 rows and 2 columns,
    # with the second column for color bars (invisible for the top plot)
    gs = GridSpec(5, 3, width_ratios=[1, 0.05, 0.3], height_ratios=[.5, 1, 0.05, .5, .5], hspace=0.2)

    # Plotting Observations on the top
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(observations[begin: begin + length], 'o-')
    ax0.axhline(0, color='black', linewidth=.5)
    ax0.set_title('Observations')
    ax0.set_ylabel('Observation Value')
    # ax0.set_ylim(min(-1.0, np.array(observations).min()), max(1.0, np.array(observations).max()))
    ax0.set_ylim(-1.0, 1.0)

    # Invisible axis for the color bar next to the observations plot
    cax0 = fig.add_subplot(gs[0, 1:])
    cax0.axis('off')
    # print(activations_matrix.shape)
    # print(activations_matrix[begin: begin + length, :].shape)

    # Plotting Activations on the bottom
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    im = ax1.imshow(activations_matrix[begin: begin + length, :].T, aspect='auto', cmap='viridis')
    if algo == 'ppo':
        ax1.set_title('Activations of action_net')
    else:
        ax1.set_title('Activations of lstm_actor')
    ax1.set_ylabel('Neuron')

    # Color bar for the activations heatmap
    cax1 = fig.add_subplot(gs[1, 1])
    cbar = plt.colorbar(im, cax=cax1)
    cbar.set_label('Activation')
    c2ax2 = fig.add_subplot(gs[1, 2])
    c2ax2.axis('off')

    x, y = zip(*agent_locations[begin: begin + length])

    cax2 = fig.add_subplot(gs[2, 0])
    # Create a color bar at the bottom indicating the colors
    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=0, vmax=len(x)))
    sm.set_array([])
    cax2 = plt.gca()  # Assuming cax2 is not defined elsewhere, get current axis
    cbar = plt.colorbar(sm, cax=cax2, orientation='horizontal')
    cbar.set_label('Step', rotation=0, labelpad=5)

    # Adjust the number of ticks to 6 to match the number of labels
    num_ticks = 6
    ticks = np.linspace(0, len(x), num=num_ticks)
    cbar.set_ticks(ticks)

    # Ensure the number of labels matches the number of ticks
    labels = [str(int(tick)) for tick in ticks]
    cbar.set_ticklabels(labels)

    cax2.axis('off')

    ax2 = fig.add_subplot(gs[3, 0], sharex=ax0)
    # speed_colors_2 = np.where(np.array(speed) > 0, 'red', 'green')
    ax2.plot(speed[begin: begin + length], '-', color='red', label='Speed')
    # ax2.scatter(range(len(speed)), speed, color=speed_colors_2)
    ax2.axhline(0, color='black', linewidth=.5)
    ax2.set_title('Speed')
    ax2.set_ylim(min(-1.0, np.array(speed[begin: begin + length]).min()),
                 max(1.0, np.array(speed[begin: begin + length]).max()))
    cax2 = fig.add_subplot(gs[3:, 1:])

    # Plot the trace with a continuous line and multiple colors
    # Use a colormap to generate colors along the spectrum
    colors = plt.cm.rainbow(np.linspace(0, 1, len(x)))
    print(len(x))
    print(y)
    print(len(y))
    print(y[1])
    print(y)
    for i in range(len(x) - 1):
        print(f'test: {x[i:i + 2]}, {y[i:i + 2]}')
        if abs(x[i + 1] - x[i]) > 0.5 or abs(y[i + 1] - y[i]) > 0.5:
            continue
        cax2.plot(x[i:i + 2], y[i:i + 2], color=colors[i], linewidth=2)

    for spine in cax2.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)
    cax2.set_title('Trace')
    cax2.grid(True)
    # Set the aspect of the plot to be equal
    cax2.set_aspect('equal')

    # Remove axis for aesthetics similar to the example image
    cax2.axis('off')

    ax3 = fig.add_subplot(gs[4, 0], sharex=ax0)
    # rotation_colors = np.where(np.array(rotation) > 0, 'blue', 'yellow')
    ax3.plot(rotation[begin: begin + length], '-', color='grey', label='Rotation')
    ax3.axhline(0, color='black', linewidth=.5)
    # ax3.scatter(range(len(rotation)), rotation, color=rotation_colors)
    ax3.set_title('Rotation')
    ax3.set_ylim(min(-1.0, np.array(rotation[begin: begin + length]).min()),
                 max(1.0, np.array(rotation[begin: begin + length]).max()))
    ax3.set_xlabel('Time (Sample Number)')

    # cax3 = fig.add_subplot(gs[3, 1])
    # cax3.axis('off')

    # Hide the x-axis labels for the top plot to avoid repetition
    # plt.setp(ax0.get_xticklabels(), visible=False)

    for point in marker_points:
        # Add vertical line on the observation plot
        ax0.axvline(x=point, color='red', linestyle='--', lw=2)

        # For the heatmap, the x-coordinates are scaled to the width of the activations matrix
        # print(point, length, n_samples)
        proportion = point / n_samples
        # We then apply this proportion to the width of the matrix to get the correct x-coordinate
        x_coord = proportion * activations_matrix.T.shape[1]

        # Add vertical line on the activations heatmap
        ax1.axvline(x=x_coord, color='red', linestyle='--', lw=2)

    # plt.tight_layout()
    # plt.show()
    print(marker_points)
    if not search_food or len(marker_points) > 0:
        if plot_save_path == '':
            plt.show()
        else:
            plt.savefig(f'{plot_save_path}/model_{model_number}_activations.png', bbox_inches='tight')
    return marker_points


if __name__ == '__main__':
    # algorithm = 'ppo'
    algorithm = 'ppo_lstm'
    # model_numbers = list(range(195, 198))
    model_numbers = [1]

    # model_numbers = [161]
    samples = 2000

    search_food = True
    load_best_model = True

    errors = []
    for model_number in model_numbers:
        # try:
        while True:
            print(f'model_number: {model_number}')
            kw_args, hint = extract_kwargs(get_model_info_file_name(algorithm), model_number)
            food_markers = collect_and_visualize(algo=algorithm, exp_id=model_number, env_name="WormWorld-v0",
                                                 n_samples=samples, load_best=load_best_model,
                                                 kwargs=kw_args,
                                                 label_hint=wrap_string(hint, 100), folder=get_log_directory(),
                                                 plot_save_path=get_save_plot_directory(algorithm),
                                                 csv_save_path=get_csv_directory(algorithm), search_food=search_food)
            if not search_food or len(food_markers) > 0:
                break
        # except Exception as e:
        #    print(f'error: {e}')
        #    errors.append(model_number)
    print(f'Errors in model numbers: {errors}')
