import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Folder paths
PLOT = r"res"
DATA = r"data/cleaned"
Y_TEST = r"data/cleaned/y_test.csv"

GRIDS = [30]
METHODS = ["multiquadric"]

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.facecolor": (1, 1, 1),
    "figure.facecolor": (1, 1, 1),
    "font.family": "serif",
    "font.size": 8,
    "legend.fontsize": 6,
    "legend.edgecolor": "black"
})

def generate_comparison_plots_2x3(grid, method, test_method):
    """
    Generates a 2x3 plot ONLY for y_pred (black dots) and saves it to a PDF.
    """
    pred_params_path = os.path.join(DATA, f"y_pred_{grid}_{method}_{test_method}.csv")
    y_test_path = Y_TEST

    try:
        print(f"Loading predicted data from {pred_params_path}")
        y_pred = pd.read_csv(pred_params_path, header=0)

        print(f"Loading original data from {y_test_path}")
        y_test = pd.read_csv(y_test_path, header=0)

        # Force consistent headers
        y_pred.columns = y_test.columns
        print("Replaced y_pred headers with y_test header.")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if y_pred.shape != y_test.shape:
        print(f"Dimension mismatch: y_pred shape {y_pred.shape}, y_test shape {y_test.shape}")
        return

    # Skip 3rd, 4th and 5th parameters (0-based indices 2,3,4)
    columns_to_plot = [col for i, col in enumerate(y_test.columns) if i not in (2, 3, 4)]
    first_three = columns_to_plot[:3]
    last_three = columns_to_plot[-3:]
    params_to_plot = list(first_three) + list(last_three)  # total = 6

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

    # --- 2x3 layout ---
    rows, cols = 2, 3
    fig_width_in = 15 / 2.54
    subplot_size = fig_width_in / cols
    fig_height_in = rows * subplot_size

    height_scale = 0.95  # <--- reduce this (e.g., 0.65–0.85) to shrink total height
    fig_height_in = rows * subplot_size * height_scale

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width_in, fig_height_in))
    axes = axes.flatten()

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")

    plt.subplots_adjust(
        left=0.085,
        right=0.98,
        top=1,     # slightly tighter
        bottom=0.06,  # slightly tighter (increase if x-labels get cut)
        wspace=0.6,
        hspace=0.3    # <--- reduce vertical space between rows (try 0.15–0.30)
    )

    try:
        for i, column in enumerate(params_to_plot):
            ax = axes[i]

            ax.scatter(y_test[column], y_pred[column], s=0.3, color="black")

            lims = config["y_limits"].get(column, (y_test[column].min(), y_test[column].max()))
            ax.plot([lims[0], lims[1]], [lims[0], lims[1]], color="red", linewidth=0.8)

            r2 = r2_score(y_test[column], y_pred[column])
            ax.text(
                0.05, 0.95,
                f"$R^2$: {r2:.3f}",
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="top"
            )

            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xticks(np.linspace(*ax.get_xlim(), 3))
            ax.set_yticks(np.linspace(*ax.get_ylim(), 3))

            ax.set_xlabel(f"{config['y_labels'].get(column, column)} test")
            ax.set_ylabel(f"{config['y_labels'].get(column, column)} predicted")

        plt.savefig(os.path.join(PLOT, f"y_pred_plot_2x3_{grid}_{method}_{test_method}.pdf"))
        plt.savefig(os.path.join(PLOT, f"y_pred_plot_2x3_{grid}_{method}_{test_method}.svg"), format="svg")
        plt.close()

    except Exception as e:
        print(f"Error generating plot: {e}")

# Run
for grid in GRIDS:
    for method in METHODS:
        for test_method in METHODS:
            generate_comparison_plots_2x3(grid, method, test_method)
