import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Folder paths
PLOT = r"res"
DATA = r"data/cleaned"
Y_TEST = r"data/cleaned/y_test.csv"

# Paths and configurations (modify as needed)
GRIDS = [20, 30, 40]
METHODS = ["linear", "cubic", "multiquadric"]

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

def generate_comparison_plots(grid, method, test_method):
    """
    Generates a comparison plot with subplots for selected parameters
    and saves it to a PDF file.
    
    Parameters:
        grid: Grid number
        method: Method used
        test_method: Test method used
    """
    # File paths for the predicted and original y data
    pred_params_path = os.path.join(DATA, f"y_pred_{grid}_{method}_{test_method}.csv")
    pred_params_ori_path = os.path.join(DATA, f"y_pred_ori.csv")
    y_test_path = Y_TEST  # Original test data

    try:
        # Load files with headers
        print(f"Loading predicted data from {pred_params_path}")
        y_pred = pd.read_csv(pred_params_path, header=0)  # Load with header

        print(f"Loading predicted data from {pred_params_ori_path}")
        y_pred_ori = pd.read_csv(pred_params_ori_path, header=0)  # Load with header
        
        print(f"Loading original data from {y_test_path}")
        y_test = pd.read_csv(y_test_path, header=0)  # Load with header
        
        # Replace the header of y_pred with the correct header from y_test
        y_pred.columns = y_test.columns
        y_pred_ori.columns = y_test.columns
        print("Replaced y_pred and y_pred_ori headers with y_test header.")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Ensure dimensions match
    if y_pred.shape != y_test.shape or y_pred_ori.shape != y_test.shape:
        print(f"Dimension mismatch: y_pred shape {y_pred.shape}, y_pred_ori shape {y_pred_ori.shape}, y_test shape {y_test.shape}")
        return

    # Skip 3rd, 4th and 5th parameters
    columns_to_plot = [col for i, col in enumerate(y_test.columns) if i not in (2, 3, 4)]
    num_params = len(columns_to_plot)
    first_three = columns_to_plot[:3]
    last_three = columns_to_plot[-3:]
    params_to_plot = list(first_three) + list(last_three)

    # Determine subplot grid layout
    # rows = (num_params // 3) + (1 if num_params % 3 else 0)
    # fig, axes = plt.subplots(rows, 3, figsize=(18, 6 * rows))
    # axes = axes.flatten()  # Flatten axes array for easy indexing

    fig_width_in = 15 / 2.54  # Convert cm to inches
    subplot_size = fig_width_in / 3  # Keep subplots square
    num_params = 12  # Number of subplots

    rows = (num_params // 3) + (1 if num_params % 3 else 0)
    fig_height_in = rows * subplot_size  # Maintain square aspect ratio
    fig, axes = plt.subplots(rows, 3, figsize=(fig_width_in, fig_height_in))
    axes = axes.flatten()

    # for i in range(num_params, len(axes)):  # Hide extra subplots
    #     fig.delaxes(axes[i])
    
    for ax in axes[:num_params]:  # Only set for used subplots
        ax.set_aspect("equal", adjustable="box")

    # Adjust subplot spacing
    plt.subplots_adjust(
        left=0.085,   # Adjust left margin
        right=0.98,  # Adjust right margin
        top=0.99,    # Adjust top margin
        bottom=0.045, # Adjust bottom margin
        wspace=0.6,  # Increase horizontal space between plots
        hspace=0.5   # Increase vertical space between plots
    )

    # # Adjust subplot spacing
    # plt.subplots_adjust(
    #     left=0.04,   # Adjust left margin
    #     right=1.025,  # Adjust right margin
    #     top=0.97,    # Adjust top margin
    #     bottom=0.055, # Adjust bottom margin
    #     # wspace=0.1,  # Increase horizontal space between plots
    #     hspace=0.7   # Increase vertical space between plots
    # )

    # # labels for each subplot
    # letters = [
    #     r"\textbf{(a)}", r"\textbf{(b)}", r"\textbf{(c)}",
    #     r"\textbf{(d)}", r"\textbf{(e)}", r"\textbf{(f)}",
    #     r"\textbf{(g)}", r"\textbf{(h)}", r"\textbf{(i)}",
    #     r"\textbf{(j)}", r"\textbf{(k)}", r"\textbf{(l)}"
    # ]
    # positions = [
    #     (0.084, 0.998), (0.433, 0.998), (0.78, 0.998),
    #     (0.084, 0.744), (0.433, 0.744), (0.78, 0.744),
    #     (0.084, 0.49), (0.433, 0.49), (0.78, 0.49),
    #     (0.084, 0.237), (0.433, 0.237), (0.78, 0.237)
    # ]

    # for letter, (x_pos, y_pos) in zip(letters, positions):
    #     fig.text(x_pos, y_pos, letter,
    #             verticalalignment="top", horizontalalignment="left")
        
    config = {
        "y_labels": {
            "F": r"$F$",
            "G": r"$G$",
            "N": r"$N$",
            "sigma0": r"$\sigma_{0}$",
            "k": r"$K$",
            "n": r"$n$"
        },
        "y_limits": {
            "F": (0, 1.3),
            "G": (0.1, 0.7),
            "N": (0, 9),
            "sigma0": (120, 300),
            "k": (280, 700),
            "n": (0.14, 0.3)
        }
    }

    try:        
        for idx, column in enumerate(params_to_plot):
            if idx >= 3:
                idx = idx + 3
            # First and third row: y_pred
            ax_pred = axes[idx]
            ax_pred.scatter(y_test[column], y_pred[column], s=0.3, color='black')
            # ax_pred.plot([y_test[column].min(), y_test[column].max()],
            #             [y_test[column].min(), y_test[column].max()], color='red', linewidth=0.8)
            ax_pred.plot([
                config["y_limits"].get(column, (y_test[column].min(), y_test[column].max()))[0],
                config["y_limits"].get(column, (y_test[column].min(), y_test[column].max()))[1]
            ], [
                config["y_limits"].get(column, (y_test[column].min(), y_test[column].max()))[0],
                config["y_limits"].get(column, (y_test[column].min(), y_test[column].max()))[1]
            ], color='red', linewidth=0.8)
            ax_pred.set_xlabel(f"{column} test")
            ax_pred.set_ylabel(f"{column} predicted")

            # Add R^2 and MAE as text in the top left corner
            r2 = r2_score(y_test[column], y_pred[column])
            ax_pred.text(
                0.05, 0.95,
                # f"$R^2$: {r2:.3f}\nMAE: {mae:.3f}",
                f"$R^2$: {r2:.3f}",
                transform=ax_pred.transAxes,
                fontsize=7,
                verticalalignment='top',
                # bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
            )

            ax_pred.set_aspect("equal")
            # ax_pred.locator_params(axis="x", nbins=4)
            # ax_pred.locator_params(axis="y", nbins=4)
            ax_pred.set_xlabel(f"{config['y_labels'].get(column, column)} test")
            ax_pred.set_ylabel(f"{config['y_labels'].get(column, column)} predicted")
            ax_pred.set_xlim(config["y_limits"].get(column, (y_test[column].min(), y_test[column].max())))
            ax_pred.set_xticks(np.linspace(*ax_pred.get_xlim(), 3))
            ax_pred.set_ylim(config["y_limits"].get(column, (y_test[column].min(), y_test[column].max())))
            ax_pred.set_yticks(np.linspace(*ax_pred.get_ylim(), 3))
            
            # Second and fourth row: y_pred_ori
            ax_pred_ori = axes[idx + 3]
            ax_pred_ori.scatter(y_test[column], y_pred_ori[column], s=0.3, color='blue')
            # ax_pred_ori.plot([y_test[column].min(), y_test[column].max()],
            #                 [y_test[column].min(), y_test[column].max()], color='red', linewidth=0.8)
            ax_pred_ori.plot([
                config["y_limits"].get(column, (y_test[column].min(), y_test[column].max()))[0],
                config["y_limits"].get(column, (y_test[column].min(), y_test[column].max()))[1]
            ], [
                config["y_limits"].get(column, (y_test[column].min(), y_test[column].max()))[0],
                config["y_limits"].get(column, (y_test[column].min(), y_test[column].max()))[1]
            ], color='red', linewidth=0.8)
            ax_pred_ori.set_xlabel(f"{column} test")
            ax_pred_ori.set_ylabel(f"{column} predicted ori")

            # Add R^2 and MAE as text in the top left corner
            r2_ori = r2_score(y_test[column], y_pred_ori[column])
            ax_pred_ori.text(
                0.05, 0.95,
                # f"$R^2$: {r2:.3f}\nMAE: {mae:.3f}",
                f"$R^2$: {r2_ori:.3f}",
                transform=ax_pred_ori.transAxes,
                fontsize=7,
                verticalalignment='top',
                # bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
            )

            ax_pred_ori.set_aspect("equal")
            # ax_pred_ori.locator_params(axis="x", nbins=4)
            # ax_pred_ori.locator_params(axis="y", nbins=4)
            ax_pred_ori.set_xlabel(f"{config['y_labels'].get(column, column)} test")
            ax_pred_ori.set_ylabel(f"{config['y_labels'].get(column, column)} predicted")
            ax_pred_ori.set_xlim(config["y_limits"].get(column, (y_test[column].min(), y_test[column].max())))
            ax_pred_ori.set_xticks(np.linspace(*ax_pred_ori.get_xlim(), 3))
            ax_pred_ori.set_ylim(config["y_limits"].get(column, (y_test[column].min(), y_test[column].max())))
            ax_pred_ori.set_yticks(np.linspace(*ax_pred_ori.get_ylim(), 3))
        

        # Adjust layout and save the plot
        # plt.tight_layout()
        plot_path = os.path.join(PLOT, f"y_pred_plot_{grid}_{method}_{test_method}.pdf")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Error generating plot: {e}")


# Iterate over grid, method, and test_method combinations
for grid in GRIDS:
    for method in METHODS:
        for test_method in METHODS:
            generate_comparison_plots(grid, method, test_method)
