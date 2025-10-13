"""
Tool to plot centroids and nodes.
"""

# %%
import csv
import os
import matplotlib.pyplot as plt

# Variables
CENT = r"data/raw/centroids.csv"
ND = r"data/raw/nodes.csv"
PLOT = r"res"

# Initialize lists for coordinates from both files
x_coords1, y_coords1, z_coords1 = [], [], []
x_coords2, y_coords2, z_coords2 = [], [], []

# Read the first CSV file
with open(CENT, mode='r') as file1:
    reader1 = csv.reader(file1)
    for row in reader1:
        x, y, z = map(float, row)
        x_coords1.append(x)
        y_coords1.append(y)
        z_coords1.append(z)

# Read the second CSV file with all coordinates in a single row
with open(ND, mode='r') as file2:
    reader2 = csv.reader(file2)
    row = next(reader2)  # Read the only row
    coordinates = list(map(float, row))  # Convert all values to floats
    
    # Group coordinates in sets of 3 (x, y, z)
    for i in range(0, len(coordinates), 3):
        x_coords2.append(coordinates[i])
        y_coords2.append(coordinates[i + 1])
        z_coords2.append(coordinates[i + 2])

# %% Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the first set of coordinates
ax.scatter(x_coords1, y_coords1, z_coords1, c='b', marker='.', label='centroids')

# Plot the second set of coordinates
ax.scatter(x_coords2, y_coords2, z_coords2, c='r', marker='.', label='nodes')

# Set labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Set the z-axis limits
ax.set_zlim(0, 5)  # Replace 0 and 1 with your desired min and max values for the z-axis

# Set the orientation of the plot (optional)
ax.view_init(elev=45, azim=-45)

# Show plot
plt.show()

# %% Create a 2D plot
plt.figure(figsize=(8,8))

# Plot the first set of coordinates
plt.scatter(x_coords1, y_coords1, c='b', marker='.', label='centroids')

# Plot the second set of coordinates
plt.scatter(x_coords2, y_coords2, c='r', marker='.', label='nodes')

# Ensure equal aspect ratio for the plot
plt.gca().set_aspect('equal', adjustable='box')

# Create a 2x2 grid by setting major ticks at 0, 15, and 30
plt.xticks([0, 5, 10, 15, 20, 25, 30])
plt.yticks([0, 5, 10, 15, 20, 25, 30])
plt.grid(True)

# Set labels and legend
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.legend()

# saves plot to external file
plt.savefig(os.path.join(PLOT, "centroids_and_nodes.pdf"))

# Show plot
plt.show()
