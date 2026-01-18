import numpy as np
import csv
import os
import matplotlib.pyplot as plt

# Variables
DATA = r"data/cleaned"
# INT_P = r"data/processed/dic_subset_coords_cent.csv"
# LEFT_PLOT = r"data/cleaned/x_test_30_multiquadric.csv"
RIGHT_PLOT = r"data/cleaned/x_test_dic_30_multiquadric.csv"
PLOT = r"res"

GRIDS = [30]
METHODS = ["multiquadric"]

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

# # Importing centroid coordinates
# x_centroids, y_centroids = [], []
# with open(INT_P, 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         x_centroids.append(float(row[0]))
#         y_centroids.append(float(row[1]))

# x_centroids = np.array(x_centroids)
# y_centroids = np.array(y_centroids)

# Original exx, eyy and exy values
x_ori_exx = []
x_ori_eyy = []
x_ori_exy = []
with open(RIGHT_PLOT, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)
    row = next(reader)
    for i in range(len(row) - (int(len(row) / 20) - 2), len(row), 3):
        x_ori_exx.append(float(row[i]))
        x_ori_eyy.append(float(row[i+1]))
        x_ori_exy.append(float(row[i+2]))
x_ori_exx = np.array(x_ori_exx)
x_ori_eyy = np.array(x_ori_eyy)
x_ori_exy = np.array(x_ori_exy)

# Determine global color scale
global_min, global_max = x_ori_exx.min(), x_ori_exx.max()

# Iterate over grid-method combinations
for grid in GRIDS:
    x_coords, y_coords = mesh_gen(grid)
    for method in METHODS:
        interpolated_file = f"x_test_{grid}_{method}.csv"

        # Interpolated exx, eyy and exy values
        x_int_exx = []
        x_int_eyy = []
        x_int_exy = []
        with open(os.path.join(DATA, interpolated_file), 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            row = next(reader)
            for i in range(len(row) - (int(len(row) / 20) - 2), len(row), 3):
                x_int_exx.append(float(row[i]))
                x_int_eyy.append(float(row[i+1]))
                x_int_exy.append(float(row[i+2]))
        x_int_exx = np.array(x_int_exx)
        x_int_eyy = np.array(x_int_eyy)
        x_int_exy = np.array(x_int_exy)

        # Update global color scale
        global_min = min(global_min, np.nanmin(x_int_exx))
        global_max = max(global_max, np.nanmax(x_int_exx))
        
        rows, cols = 3, 2

        fig_width_in = 13.8 / 2.54  # Convert cm to inches
        subplot_size = fig_width_in / 2  # Keep subplots square
        num_params = 6  # Number of subplots

        fig_height_in = rows * subplot_size  # Maintain square aspect ratio
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width_in, fig_height_in))
        # axes = axes.flatten()

        # for ax in axes[:num_params]:  # Only set for used subplots
        #     ax.set_aspect("equal", adjustable="box")
        
        # Adjust subplot spacing
        plt.subplots_adjust(
            left=0.08,   # Adjust left margin
            right=0.985,  # Adjust right margin
            top=0.99,    # Adjust top margin
            bottom=0.1, # Adjust bottom margin
            wspace=0.7,  # Increase horizontal space between plots
            # hspace=0   # Increase vertical space between plots
        )

        # labels for each subplot
        letters = [
            r"\textbf{(a)}", r"\textbf{(b)}",
            r"\textbf{(c)}", r"\textbf{(d)}",
            r"\textbf{(e)}", r"\textbf{(f)}"
        ]
        positions = [
            (0.08, 0.995), (0.653, 0.995),
            (0.08, 0.683), (0.653, 0.683),
            (0.08, 0.37), (0.653, 0.37)
        ]

        for letter, (x_pos, y_pos) in zip(letters, positions):
            fig.text(x_pos, y_pos, letter,
                    verticalalignment="top", horizontalalignment="left")

        # Labels for subplots
        titles = [r"$\varepsilon_{xx,", r"$\varepsilon_{yy,", r"$\varepsilon_{xy,"]

        # define element size according to grid
        match grid:
            case 20:
                size = 63
            case 30:
                size = 34
            case 40:
                size = 17
            case _:
                size = 4

        # Compute a global vmin and vmax
        global_min = min(np.nanmin(x_int_exx), np.nanmin(x_int_eyy), np.nanmin(x_int_exy),
                        np.nanmin(x_ori_exx), np.nanmin(x_ori_eyy), np.nanmin(x_ori_exy))
        global_max = max(np.nanmax(x_int_exx), np.nanmax(x_int_eyy), np.nanmax(x_int_exy),
                        np.nanmax(x_ori_exx), np.nanmax(x_ori_eyy), np.nanmax(x_ori_exy))
        
        # print(global_min, global_max)

        # Define the data pairs for each row
        data_pairs = [
            (x_int_exx, x_ori_exx),
            (x_int_eyy, x_ori_eyy),
            (x_int_exy, x_ori_exy)
        ]

        # Define consistent axis limits and tick intervals
        x_min, x_max = 0, 30
        y_min, y_max = 0, 30
        tick_interval = 5
        x_ticks = np.arange(x_min, x_max + tick_interval, tick_interval)
        y_ticks = np.arange(y_min, y_max + tick_interval, tick_interval)

        # Iterate over each row (exx, eyy, exy)
        for row_idx, (x_int_left, x_int_right) in enumerate(data_pairs):
            # Compute vmin and vmax for this row separately
            # vmin_row = min(np.nanmin(x_int), np.nanmin(x_ori))
            # vmax_row = max(np.nanmax(x_int), np.nanmax(x_ori))

            # Left column (interpolated, original)
            scatter_int_left = axes[row_idx, 0].scatter(
                x_coords, y_coords, c=x_int_left, cmap='jet', s=size, alpha=0.9, edgecolors='none',
                vmin=global_min, vmax=global_max
            )
            # axes[row_idx, 0].set_title(f'{titles[row_idx]} - Interpolated')
            axes[row_idx, 0].set_xlabel('x [mm]')
            axes[row_idx, 0].set_ylabel('y [mm]')
            axes[row_idx, 0].set_aspect("equal", adjustable="box")

            # equal scales and ticks on both axes
            axes[row_idx, 0].set_xlim(x_min, x_max)
            axes[row_idx, 0].set_ylim(y_min, y_max)
            axes[row_idx, 0].set_xticks(x_ticks)
            axes[row_idx, 0].set_yticks(y_ticks)
            axes[row_idx, 0].set_aspect("equal")  # force equal scaling

            # Right column (interpolated, after DIC filtering)
            scatter_int_right = axes[row_idx, 1].scatter(
                x_coords, y_coords, c=x_int_right, cmap='jet', s=size, alpha=0.9, edgecolors='none',
                vmin=global_min, vmax=global_max
            )
            # axes[row_idx, 0].set_title(f'{titles[row_idx]} - Interpolated')
            axes[row_idx, 1].set_xlabel('x [mm]')
            axes[row_idx, 1].set_ylabel('y [mm]')
            axes[row_idx, 1].set_aspect("equal", adjustable="box")

            # equal scales and ticks on both axes
            axes[row_idx, 1].set_xlim(x_min, x_max)
            axes[row_idx, 1].set_ylim(y_min, y_max)
            axes[row_idx, 1].set_xticks(x_ticks)
            axes[row_idx, 1].set_yticks(y_ticks)
            axes[row_idx, 1].set_aspect("equal")  # force equal scaling

            # # Right column (original)
            # scatter_ori = axes[row_idx, 1].scatter(
            #     x_centroids, y_centroids, c=x_ori, cmap='jet', s=1, alpha=0.9, edgecolors='none',
            #     vmin=global_min, vmax=global_max, rasterized=True
            # )
            # # axes[row_idx, 1].set_title(f'{titles[row_idx]} - Original')
            # axes[row_idx, 1].set_xlabel('x [mm]')
            # axes[row_idx, 1].set_ylabel('y [mm]')
            # axes[row_idx, 1].set_aspect("equal", adjustable="box")

            # # equal scales and ticks on both axes
            # axes[row_idx, 1].set_xlim(x_min, x_max)
            # axes[row_idx, 1].set_ylim(y_min, y_max)
            # axes[row_idx, 1].set_xticks(x_ticks)
            # axes[row_idx, 1].set_yticks(y_ticks)
            # axes[row_idx, 1].set_aspect("equal")  # force equal scaling

            # Add a colorbar for this row
            # cbar = fig.colorbar(scatter_ori, ax=axes[row_idx, :].ravel().tolist(), label=titles[row_idx])

        # Add a single horizontal colorbar at the bottom
        cax = fig.add_axes([0.2, 0.03, 0.6, 0.02])  # [left, bottom, width, height]
        cbar = fig.colorbar(scatter_int_right, cax=cax, orientation='horizontal')
        # cbar.set_label(r'$\varepsilon$')
        ticks = np.linspace(global_min, global_max, num=7)
        cbar.set_ticks(ticks)

        # saves and closes plot
        plt.savefig(os.path.join(PLOT, f'interp_{grid}_{method}_.pdf'), dpi=300)
        plt.close()
