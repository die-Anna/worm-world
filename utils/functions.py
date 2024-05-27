import json
import os
import re
from pathlib import Path

from stable_baselines3.common.monitor import Monitor

root_dir = '.'


def extract_kwargs(file_name, key):
    with open(file_name, 'r') as file:
        data = json.load(file)

    # The key you're interested in
    key = str(key)

    # Extract the string associated with the key
    model_info = data.get(key)
    if not model_info:
        print("model info not found")
        raise ModelInfoFoundError(f"Model info for model {key} not found in file {file_name}")

    # Find the substring that starts with "kwargs="
    hint_string = model_info.split("kwargs=")
    kwargs_str = hint_string[-1].strip()
    # Use regex to match key-value pairs, taking quoted strings into account
    pattern = re.compile(r"(\w+):(?:'([^']*)'|(.*?)(?=\s+\w+:|$))")
    # Build the dictionary
    kwargs = {}
    for match in pattern.finditer(kwargs_str):
        key = match.group(1)
        # Use the appropriate group based on whether it's quoted or not
        value = match.group(2) or match.group(3).strip()
        # Attempt to interpret numeric values correctly
        if value.isdigit():
            value = int(value)  # Convert digits to integers
        elif value.replace('.', '', 1).isdigit():
            value = float(value)  # Convert float representations
        elif value == 'True':
            value = True
        elif value == 'False':
            value = False
        kwargs[key] = value

    return kwargs, model_info


class ModelInfoFoundError(Exception):
    """
    Raised when loading the Model loading fails.
    """

    pass


def get_model_directory(algorithm, model_no):
    return f'{root_dir}/logs/{algorithm}/WormWorld-v0_{model_no}/'


def get_algorithm_log_directory(algorithm):
    return f'{root_dir}/logs/{algorithm}/'


def get_log_directory():
    return f'{root_dir}/logs'


def get_model_info_file_name(algorithm):
    if not os.path.exists(f'{root_dir}/data'):
        os.makedirs(f'{root_dir}/data')
    return f'{root_dir}/data/trained_model_infos_{algorithm}.json'


def get_save_plot_directory(algorithm):
    directory = f'{root_dir}/data/{algorithm}'
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def get_csv_directory(algorithm):
    directory = f'{root_dir}/data/{algorithm}/csv'
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def get_save_video_directory(algorithm):
    directory = f'{root_dir}/data/{algorithm}/video'
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def get_hyperparams_directory():
    return f'{root_dir}/data'


def get_value(file_path, key):
    """Get the string value associated with the given integer key."""
    data = read_data(file_path)
    return data.get(str(key))  # Convert the key to a string as keys are stored as strings in JSON


def write_data(file_path, data):
    """Write data to a JSON file."""
    with open(file_path, 'w+') as file:
        json.dump(data, file, indent=4)


def set_value(file_path, key, value):
    """Set the string value for the given integer key."""
    data = read_data(file_path)
    data[str(key)] = value  # Convert the key to a string to use it in JSON
    write_data(file_path, data)


def read_data(file_path):
    """Read data from a JSON file, returning an empty dictionary if the file does not exist."""
    if os.path.exists(file_path):
        with open(file_path, 'r+') as file:
            return json.load(file)
    else:
        return {}


def wrap_string(s, max_width):
    words = s.split()
    lines = []
    current_line = ""

    for word in words:
        # Check if adding the next word would exceed the max width
        if len(current_line) + len(word) + 1 > max_width:
            lines.append(current_line)
            current_line = word
        else:
            if current_line:
                # Add a space before the word if it's not the start of a line
                current_line += " "
            current_line += word

    # Don't forget to add the last line
    lines.append(current_line)

    return '\n'.join(lines)
