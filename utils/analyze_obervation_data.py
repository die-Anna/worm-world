from functions import *
import pandas as pd
import numpy as np
import ast

# disable oneDNN optimizations to avoid floating-point discrepancies
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model_no = 220
algorithm = 'ppo'

# Load the data from the CSV file
file_path = get_csv_directory(algorithm) + f'/data_model_long_{model_no}.csv'

data = pd.read_csv(file_path)
data.columns = ['timestep', 'speed', 'observation', 'rotation']  # add timestep column

# Convert the string representation of lists into actual lists
data['observation'] = data['observation'].apply(ast.literal_eval)

# Extract numeric values from these lists
data['observation_value'] = data['observation'].apply(lambda x: np.array(x).item())

# Display statistics and distribution information
print(data['observation_value'].describe(), data['observation_value'].hist())
