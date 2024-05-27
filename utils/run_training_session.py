import subprocess
import sys
from functions import *
from my_config import *
import os


def get_max_number_from_folder_names(folder_path, pattern="WormWorld-v0_"):
    max_number = 0
    # Compile the regex pattern to match folder names and extract the number
    regex = re.compile(rf"{pattern}(\d+)")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
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
    optimize = False
    # when the following two params are set to False LSTM is used
    train_hopfield = False
    use_ppo_frame_stacking = False

    total_time_steps = 7_000_000
    beta = 2.0
    sigma = 3.0
    min_number_food_objects = 2
    max_number_food_objects = 4
    reward_type = 'math'
    # reward_type = 'physics'
    test_eat_fixed_amount = True
    hunger_penalty = True
    gradient_reward = True

    use_multiplicative_noise = True
    use_additive_noise = True

    decay = False
    decay_rate = 0.003
    food_memory = False

    frame_stacking = True
    frame_stack_size = 8
    decay_factor = 200

    algorithm = "ppo_lstm" if not train_hopfield else "ppo"
    study_name = 'optimize-lstm-new' if not train_hopfield else 'optimize-hopfield_new'

    hyperparams_file = f'{get_hyperparams_directory()}/hyperparams_lstm_optimized.yml' if not train_hopfield \
        else f'{get_hyperparams_directory()}/hyperparams_hopfield.yml'

    if use_ppo_frame_stacking:
        study_name = 'optimize-ppo-frame-stacking-2'
        frame_stacking = True
        algorithm = 'ppo'
        hyperparams_file = f'{get_hyperparams_directory()}/hyperparams_ppo_fs_optimized.yml'
    print(f'{algorithm} - {hyperparams_file}')

    info_file_path = get_model_info_file_name(algorithm)
    # Constructing the arguments as a JSON string

    script_file = './train.py'
    env_kwargs = (f"beta:{beta} reward_type:'{reward_type}' min_number_food_objects:{min_number_food_objects} "
                  f"max_number_food_objects:{max_number_food_objects} test_eat_fixed_amount:{test_eat_fixed_amount} "
                  f"hunger_penalty:{hunger_penalty} gradient_reward:{gradient_reward} "
                  f"sigma:{sigma} "
                  f"frame_stacking:{frame_stacking} frame_stack_size:{frame_stack_size} "
                  f"use_multiplicative_noise:{use_multiplicative_noise} use_additive_noise:{use_additive_noise} "
                  f"food_memory:{food_memory} decay:{decay} decay_rate:{decay_rate} decay_factor:{decay_factor}"
                  )

    # Ensure this is treated as a single argument in the command list
    if optimize:
        cmd = [
            sys.executable, script_file,
            '--algo', algorithm,
            '--env', 'WormWorld-v0',
            '-n', str(total_time_steps),
            '-f', f'{get_log_directory()}',
            '--conf', hyperparams_file,
            '--env-kwargs', *env_kwargs.split(' '),
            '-optimize',
            '--study-name', study_name,
            '--storage', "mysql://" + mysql_user + ":" + mysql_password + "/worms"
        ]
    else:
        cmd = [
            sys.executable, script_file,
            '--algo', algorithm,
            '--env', 'WormWorld-v0',
            '-n', str(total_time_steps),
            '-f', f'{get_log_directory()}',
            '--conf', hyperparams_file,
            '--env-kwargs', *env_kwargs.split(' '),
        ]

    # Use Popen for real-time output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    model_no = get_max_number_from_folder_names(get_algorithm_log_directory(algorithm))
    model_values = (f"conf-file={hyperparams_file},time_steps={total_time_steps}, beta={beta}, "
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
