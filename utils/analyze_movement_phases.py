from functions import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# disable oneDNN optimizations to avoid floating-point discrepancies
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model_numbers = [1]
algorithm = 'ppo_lstm'
window_size = 80
threshold = 0.9

analyze_gd = False  # set to True if chemotaxis algorithm should be tested
if analyze_gd:
    window_size = 250
    threshold = 0.92

for model_no in model_numbers:
    # Load the data from the CSV file
    if analyze_gd:
        file_path = get_csv_directory('gd') + f'/data_model_long_gd.csv'
    else:
        file_path = get_csv_directory(algorithm) + f'/data_model_long_{model_no}.csv'

    data = pd.read_csv(file_path)
    data.columns = ['timestep', 'speed', 'observation', 'rotation']  # add timestep column

    # Standardize the 'speed' and 'rotation' columns
    data['speed'] = (data['speed'] - data['speed'].mean()) / data['speed'].std()
    data['rotation'] = (data['rotation'] - data['rotation'].mean()) / data['rotation'].std()

    # Compute rolling standard deviation over a window
    data['speed_std'] = data['speed'].rolling(window=window_size, min_periods=1, center=True).std()
    data['rotation_std'] = data['rotation'].rolling(window=window_size, min_periods=1, center=True).std()

    # Print the rolling standard deviation of 'speed'
    print(data[['speed_std', 'rotation_std']])

    # Combine by taking the maximum of the two standard deviations
    data['combined_std'] = np.maximum(data['speed_std'], data['rotation_std'])

    # Optionally, you can save or display the modified data
    print(data[['combined_std']])

    # Define threshold for 'constant' phases based on combined standard deviation
    combined_threshold = data['combined_std'].quantile(threshold)  # threshold

    # Determine constant phases
    data['constant_phase'] = (data['combined_std'] <= combined_threshold).astype(int)

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Speed plot
    axes[0].plot(data['timestep'], data['speed'], label='Speed', color='blue')
    axes[0].fill_between(data['timestep'], data['speed'].min(), data['speed'].max(),
                         where=data['constant_phase'] == 1, color='grey', alpha=0.5, interpolate=True)
    axes[0].set_title('Speed with Global Search Phases Highlighted')
    axes[0].set_ylabel('Speed')
    axes[0].set_xlabel('Timestep')

    # Rotation plot
    axes[1].plot(data['timestep'], data['rotation'], label='Rotation', color='red')
    axes[1].fill_between(data['timestep'], data['rotation'].min(), data['rotation'].max(),
                         where=data['constant_phase'] == 1, color='grey', alpha=0.5, interpolate=True)
    axes[1].set_title('Rotation with Global Search Phases Highlighted')
    axes[1].set_ylabel('Rotation')
    axes[1].set_xlabel('Timestep')

    plt.tight_layout()
    # plt.show()
    if analyze_gd:
        plt.savefig(f'{get_save_plot_directory("gd")}/model_gd_movement_phases.png', bbox_inches='tight')
    else:
        plt.savefig(f'{get_save_plot_directory(algorithm)}/model_{model_no}_movement_phases.png', bbox_inches='tight')
