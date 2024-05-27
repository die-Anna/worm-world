import itertools
import subprocess
import sys
from functions import *


def get_max_number_from_folder_names(folder_path, pattern="WormWorld-v0_"):
    max_number = -1
    # Compile the regex pattern to match folder names and extract the number
    regex = re.compile(rf"{pattern}(\d+)")

    # List all items in the given folder path
    for item in os.listdir(folder_path):
        # Check if the item is a folder and matches the pattern
          if os.path.isdir(os.path.join(folder_path, item)):
            match = regex.match(item)
            if match:
                # Extract the number part and compare
                number = int(match.group(1))
                if number > max_number:
                    max_number = number
    return max_number + 1


if __name__ == '__main__':

    train_hopfield = False
    total_time_steps = 2_000_000

    min_number_food_objects = 2
    max_number_food_objects = 5

    test_eat_fixed_amount = True
    hunger_penalty = True
    gradient_reward = True

    frame_stacking = False
    frame_stack_size = 8

    param_grid = {
        'reward_type': ['math', 'physics'],
        'noise_std_dev': [0.001, 0.003, 0.005],
        'sigma': [2, 4, 6],
        'beta': [1.0, 2.0, 3.0]
    }

    all_params = list(itertools.product(*param_grid.values()))

    for params in all_params:
        print(params)
        algorithm = "ppo_lstm" if not train_hopfield else "ppo"
        # config_file = "config.py" if not train_hopfield else "config_hl.py"
        info_file_path = get_model_info_file_name(algorithm)
        hyperparams_file = f'{get_hyperparams_directory()}/hyperparams_lstm_optimized.yml' if not train_hopfield \
            else f'{get_hyperparams_directory()}/hyperparams_hopfield.yml'

        script_file = '../train.py'
        env_kwargs = (f"beta:{params[3]} reward_type:'{params[0]}' min_number_food_objects:{min_number_food_objects} "
                      f"max_number_food_objects:{max_number_food_objects} test_eat_fixed_amount:{test_eat_fixed_amount} "
                      f"hunger_penalty:{hunger_penalty} gradient_reward:{gradient_reward} "
                      f"sigma:{params[2]} noise_std_dev:{params[1]}"
                      # f"frame_stacking:{frame_stacking} frame_stack_size:{frame_stack_size} "
                      )

        # Ensure this is treated as a single argument in the command list

        cmd = [
            sys.executable, script_file,
            '--algo', algorithm,
            '--env', 'WormWorld-v0',
            '-n', str(total_time_steps),
            '-f', f'{get_log_directory()}',
            # '--hyperparams-file', 'hyperparams_hl.yml',
            # '--conf', config_file,
            '--conf', hyperparams_file,
            '--env-kwargs', *env_kwargs.split(' '),
        ]

        # Use Popen for real-time output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        model_no = get_max_number_from_folder_names(get_algorithm_log_directory(algorithm))
        model_values = (f"conf-file={hyperparams_file},time_steps={total_time_steps}, "
                        f"bumps=({min_number_food_objects}-{max_number_food_objects}) kwargs={env_kwargs}")
        set_value(info_file_path, model_no, model_values)
        # Stream the output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        # Check the return code
        rc = process.poll()
        if rc == 0:
            print("Script executed successfully")
        else:
            print(f"Script returned with error code: {rc}")
