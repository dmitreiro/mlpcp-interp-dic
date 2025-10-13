"""
Tool to plot centroids and nodes.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

DIC_CENTROIDS = r"res/dic_mesh_centroids.pdf"
CENT = r"data/raw/dic_centroids.csv"
MESH = r"res/cruciform_mesh_refined.png"

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.facecolor": (1,1,1),
    "figure.facecolor": (1,1,1),
    # "font.family": "serif",
    "font.family": "Palatino",
    "font.size": 8,
    "legend.fontsize": 6,
    "legend.edgecolor": "black"
})

# Define figure size
fig_width_in = 13.8 / 2.54  # Convert cm to inches
subplot_size = fig_width_in / 2  # Each subplot should be square
fig_height_in = fig_width_in / 2  # Keep aspect ratio square

# Set up the plotting grid
fig, axes = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in), sharey=False)

# Assign subplots
ax1, ax_empty = axes.flatten()

# Read and show the image
img = plt.imread(MESH)
ax_empty.imshow(img)
ax_empty.axis('off')  # Optional: hides the axis lines and ticks
ax_empty.set_aspect("equal")  # Ensures image aspect ratio matches the plot

# Adjust subplot spacing
plt.subplots_adjust(
    left=0.03,
    right=1.03,
    top=0.88,
    bottom=0.14,
    # wspace=0.35
)

# labels for each subplot
letters = [r"\textbf{(a)}", r"\textbf{(b)}"]
positions = [(0.07, 0.97), (0.62, 0.97)]

for letter, (x_pos, y_pos) in zip(letters, positions):
    fig.text(x_pos, y_pos, letter,
            verticalalignment="top", horizontalalignment="left")

# Initialize lists for coordinates from both files
x_coords1, y_coords1, z_coords1 = [], [], []

# Define consistent axis limits and tick intervals
x_min, x_max = 0, 30
y_min, y_max = 0, 30
tick_interval = 5
x_ticks = np.arange(x_min, x_max + tick_interval, tick_interval)
y_ticks = np.arange(y_min, y_max + tick_interval, tick_interval)

# Read the first CSV file
with open(CENT, mode='r') as file1:
    reader1 = csv.reader(file1)
    for row in reader1:
        x, y, z = map(float, row)
        x_coords1.append(x)
        y_coords1.append(y)
        z_coords1.append(z)

# Plot the first set of coordinates
# axes[0].scatter(x_coords1, y_coords1, c='r', marker='.', s=1.2, label='Centroids')
axes[0].scatter(x_coords1, y_coords1, c='r', marker='.', s=1.2)
axes[0].set_xlabel('x (mm)')
axes[0].set_ylabel('y (mm)')
# axes[0].legend(fontsize=7, markerscale=1, labelspacing=0.05, handletextpad=0.5, loc="upper right")

# equal scales and ticks on both axes
axes[0].set_xlim(x_min, x_max)
axes[0].set_ylim(y_min, y_max)
axes[0].set_xticks(x_ticks)
axes[0].set_yticks(y_ticks)
axes[0].set_aspect("equal")  # force equal scaling

# saves plot to external file
plt.savefig(DIC_CENTROIDS, dpi=600)

# Show plot
# plt.show()
