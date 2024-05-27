import optuna
import sqlalchemy
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_contour
from utils.my_config import *

# Define the database URL
database_url = 'mysql://' + mysql_user + ':' + mysql_password + '/worms'

# Connect to the database
engine = sqlalchemy.create_engine(database_url)

# Load the Optuna study from the database
study_name = 'optimize-ppo-frame-stacking-2'  # Replace 'your_study_name' with the name of your study
# study_name = 'optimize-lstm-new'  # Replace 'your_study_name' with the name of your study
study = optuna.study.load_study(study_name, storage=database_url)

# Plot optimization history
plot_optimization_history(study).show()

# Plot parameter importances
plot_param_importances(study).show()

# Plot slice
plot_slice(study).show()

# Plot contour
plot_contour(study).show()

# Get the best parameters and the best value
best_params = study.best_params
best_value = study.best_value

print("Best parameters:", best_params)
print("Best value:", best_value)
