import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
import matplotlib.cm as cm

'''
The data was kindly provided by the Neurobiology Institute Vienna
'''


# Create a DataFrame
df = pd.read_csv('data_real_worm.txt')

# Convert time to datetime and then to seconds
df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S.%f')
start_time = df['time'].iloc[0]
df['time_seconds'] = (df['time'] - start_time).dt.total_seconds()

# Normalize time for color mapping
norm = Normalize(vmin=df['time_seconds'].min(), vmax=df['time_seconds'].max())
cmap = cm.viridis

fig, ax = plt.subplots(figsize=(10, 6))

for i in range(len(df) - 1):
    ax.plot(df['X'][i:i+2], df['Y'][i:i+2], color=cmap(norm(df['time_seconds'][i])))

# Mark the first and last points
ax.scatter(df['X'].iloc[0], df['Y'].iloc[0], color='red', label='Start', zorder=5)
ax.scatter(df['X'].iloc[-1], df['Y'].iloc[-1], color='blue', label='End', zorder=5)

# Create a mappable object for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # only needed for matplotlib < 3.1

cbar = plt.colorbar(sm, ax=ax, label='Time (seconds)')

ax.set_xlabel('Distance (mm)')
ax.set_ylabel('Distance (mm)')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
