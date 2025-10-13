# Importing libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Variables
CENT = r"data/raw/centroids.csv"
PLOT = r"res"

# path to save plots
GRIDS_COMPILATION = os.path.join(PLOT, "grids_compilation.pdf")

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

# Domain region plot
def mesh_gen(n_points: int):
    """
    Function to define mesh grid for interpolation.
    The mesh is filtered to fit cruciform geometry domain.
    """
    # Define the grid
    x = np.linspace(0, 30, n_points)
    y = np.linspace(0, 30, n_points)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.flatten(), yy.flatten()])

    # Define the conditions for the region
    in_main_square = (points[:, 0] >= 0) & (points[:, 0] <= 30) & (points[:, 1] >= 0) & (points[:, 1] <= 30)
    out_excluded_square = ~((points[:, 0] > 15) & (points[:, 0] <= 30) & (points[:, 1] > 15) & (points[:, 1] <= 30))
    out_excluded_circle = ((points[:, 0] - 15)**2 + (points[:, 1] - 15)**2) >= 7**2
    in_fillet_circ_1 = ((points[:, 0] - 12.5)**2 + (points[:, 1] - 24.17)**2) <= 2.5**2
    in_fillet_circ_2 = ((points[:, 0] - 24.17)**2 + (points[:, 1] - 12.5)**2) <= 2.5**2
    out_square_1 = (points[:, 0] > 13.16) & (points[:, 0] < 15) & (points[:, 1] > 21.75) & (points[:, 1] < 24.17)
    out_square_2 = (points[:, 0] > 21.75) & (points[:, 0] < 24.17) & (points[:, 1] > 13.16) & (points[:, 1] < 15)

    # Keep points only within the circles for these squares
    square_1_cond = out_square_1 & in_fillet_circ_1
    square_2_cond = out_square_2 & in_fillet_circ_2

    # Combine all conditions
    final_region = (
        in_main_square
        & out_excluded_square
        & out_excluded_circle
        & ~out_square_1
        & ~out_square_2
    )
    final_region |= square_1_cond | square_2_cond

    # Extract the valid points
    valid_points = points[final_region]

    # Separate into x and y coordinates
    x_coords = valid_points[:, 0]  # All rows, first column
    y_coords = valid_points[:, 1]  # All rows, second column
    
    return x_coords, y_coords

# Load the centroids
centroids = pd.read_csv(CENT, header=None)  # Assuming no header
centroid_x = centroids.iloc[:, 0]  # First column
centroid_y = centroids.iloc[:, 1]  # Second column

# Generate the coordinates for different grid sizes
grid_sizes = [20, 30, 40]
coords = [mesh_gen(n_points) for n_points in grid_sizes]

# print total points vs points inside domain
print(
    f"Mesh grid 20: {len(coords[0][0])}/{grid_sizes[0]*grid_sizes[0]} "
    f"points ({(len(coords[0][0])/(grid_sizes[0]*grid_sizes[0]))*100}%)"
)
print(
    f"Mesh grid 30: {len(coords[1][0])}/{grid_sizes[1]*grid_sizes[1]} "
    f"points ({(len(coords[1][0])/(grid_sizes[1]*grid_sizes[1]))*100}%)"
)
print(
    f"Mesh grid 40: {len(coords[2][0])}/{grid_sizes[2]*grid_sizes[2]} "
    f"points ({(len(coords[2][0])/(grid_sizes[2]*grid_sizes[2]))*100}%)"
)

# Define figure size
fig_width_in = 13.8 / 2.54  # Convert cm to inches
subplot_size = fig_width_in / 2  # Each subplot should be square
fig_height_in = subplot_size * 2  # Ensuring square aspect ratio

# Create the figure with a 2-row, 2-column grid (all plots are the same size)
fig, axes = plt.subplots(2, 2, figsize=(fig_width_in, fig_height_in),
                         gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})

# Assign subplots
ax1, ax2, ax3, ax_empty = axes.flatten()

# Remove the empty subplot (bottom-right)
fig.delaxes(ax_empty)

# List of active axes
plot_axes = [ax1, ax2, ax3]

# Adjust subplot spacing
plt.subplots_adjust(
    left=0.04,
    right=1.02,
    top=0.96,
    bottom=0.08,
    hspace=0.35,
    #wspace=-0.1
)

# labels for each subplot
letters = [r"\textbf{(a)}", r"\textbf{(b)}",
           r"\textbf{(c)}"
           ]
positions = [(0.08, 0.995), (0.615, 0.995),
             (0.08, 0.49)
             ]

for letter, (x_pos, y_pos) in zip(letters, positions):
    fig.text(x_pos, y_pos, letter,
             verticalalignment="top", horizontalalignment="left")

# Define consistent axis limits and tick intervals
x_min, x_max = 0, 30
y_min, y_max = 0, 30
tick_interval = 5
x_ticks = np.arange(x_min, x_max + tick_interval, tick_interval)
y_ticks = np.arange(y_min, y_max + tick_interval, tick_interval)

# Plot each grid
for ax, letter, (n_points, (x_coords, y_coords)) in zip(plot_axes, letters, zip(grid_sizes, coords)):
    ax.scatter(x_coords, y_coords, color="blue", s=0.3, label="Grid") # grid points
    ax.scatter(centroid_x, centroid_y, color="red", s=0.3, label="Centroids") # centroids
    # ax.set_title(f"Grid: {n_points}x{n_points}")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.legend(fontsize=7, markerscale=1, labelspacing=0.05, handletextpad=0.5, loc="upper right")
    #ax.grid(True)
    
    # equal scales and ticks on both axes
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_aspect("equal")  # force equal scaling

# Adjust layout and show the combined plots
# plt.tight_layout()

# saves plot to external file
plt.savefig(GRIDS_COMPILATION, dpi=300)

# Now display each subplot individually
for i, (n_points, (x_coords, y_coords)) in enumerate(zip(grid_sizes, coords), start=1):
    plt.figure(figsize=(8, 8))  # Create a new figure for each individual plot
    # Scatter plot for the mesh grid points
    plt.scatter(x_coords, y_coords, color="blue", s=2, label="Grid")
    # Scatter plot for the centroids
    plt.scatter(centroid_x, centroid_y, color="red", s=2, label="Centroids")
    plt.title(f"Grid: {n_points}x{n_points}")
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.legend()  # Add a legend
    plt.grid(True)

    # saves plot to external file
    plt.savefig(os.path.join(PLOT, f"grid_{n_points}x{n_points}.pdf"))
