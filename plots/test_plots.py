import os
import inspect
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import FuncFormatter

# Variables
TST_METRICS = r"metrics/testing_performance_metrics.csv"
TRAIN_METRICS = r"metrics/training_performance_metrics.csv"
REV_INTERP_METRICS = r"metrics/reverse_interpolation_metrics.csv"
INTERP_METRICS = r"metrics/interpolation_metrics.csv"
DIC_PREDICT_METRICS = r"metrics/dic_prediction_metrics.csv"
PLOT = r"res"
DATA = r"data/cleaned"

# path to save plots
TST_SIMPLE_BYGRID_PLOT = os.path.join(PLOT, "tst_simple_bygrid_metrics.pdf")
DIC_PREDICT_BYGRID_PLOT = os.path.join(PLOT, "dic_predict_bygrid_metrics.pdf")
TST_CROSS_BYGRID_PLOT = os.path.join(PLOT, "tst_cross_bygrid_metrics.pdf")
TST_CROSS_BYGRID_R2_PLOT = os.path.join(PLOT, "tst_cross_bygrid_r2_metrics.pdf")
REV_INTERP_BYGRID_PLOT = os.path.join(PLOT, "rev_interp_bygrid_metrics.pdf")
TIME_GRID_PLOT = os.path.join(PLOT, "time_grid_metrics.pdf")

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

def conditional_comma_formatter(x, pos):
    if abs(x) >= 10000:  # 5 or more digits
        return f'{x:,.0f}'  # Comma formatting
    else:
        return f'{x:.0f}'  # No comma


def plot_config():
    # General plot configuration
    config = {
        "y_labels": {"r2": "R²", "mae": "MAE", "mape": "MAPE"},
        "palette": sns.color_palette("tab10")[:3],
        "metrics": ["r2", "mae", "mape"],
        "titles": ["R²", "MAE", "MAPE"],
    }
    return config

def tst_simple_bygrid_plot():
    # Load the data from the CSV file
    df = pd.read_csv(TST_METRICS)

    # load config
    plt_conf = plot_config()

    # filters data
    filtered_df = df[df['model_method'] == df['test_method']]
    # filtered_df = filtered_df[df['grid'] == 20]

    # Define figure size
    fig_width_in = 13.8 / 2.54  # Convert cm to inches
    subplot_size = fig_width_in / 3  # Each subplot should be square
    fig_height_in = fig_width_in / 3  # Keep aspect ratio square

    # Set up the plotting grid
    fig, axes = plt.subplots(1, 3, figsize=(fig_width_in, fig_height_in), sharey=False)
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    # fig.canvas.manager.set_window_title(f"{inspect.stack()[0][3]}")

    # Adjust subplot spacing
    plt.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.2, wspace=0.5)

    # labels for each subplot
    letters = [r"\textbf{(a)}", r"\textbf{(b)}", r"\textbf{(c)}"]
    positions = [(0.09, 0.99), (0.427, 0.99), (0.764, 0.99)]

    for letter, (x_pos, y_pos) in zip(letters, positions):
        fig.text(x_pos, y_pos, letter,
             verticalalignment="top", horizontalalignment="left")

    y_limits = [(0.96, 1.02), (0.5, 2), (0.01, 0.02)]  # Custom y-axis limits

    # Create a plot for each metric
    for i, (ax, metric, title, ylim) in enumerate(zip(axes, plt_conf["metrics"], plt_conf["titles"], y_limits)):
        # Use seaborn's barplot for each metric
        sns.barplot(
            data=filtered_df,
            x="grid",
            y=metric,
            hue="model_method",
            ax=ax,
            palette=plt_conf["palette"]
        )
        # ax.set_title(title)
        ax.set_xlabel("Grid")
        ax.set_ylabel(plt_conf["titles"][i])
        # ax.legend(title="Method", loc="upper right")
        ax.legend(fontsize=6, markerscale=1, labelspacing=0.05, handletextpad=0.5, loc="upper right")
        ax.set_ylim(ylim)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # ax.set_aspect("equal")  # Forces equal scaling

    # Adjust layout
    # plt.tight_layout()

    # saves plot to external file
    plt.savefig(TST_SIMPLE_BYGRID_PLOT)

    # shows plot
    # plt.show()

def dic_predict_bygrid_plot():
    # Load the data from the CSV file
    df = pd.read_csv(DIC_PREDICT_METRICS)

    # load config
    plt_conf = plot_config()

    # filters data
    filtered_df = df[df['model_method'] == df['test_method']]
    # filtered_df = filtered_df[df['grid'] == 20]

    # Define figure size
    fig_width_in = 13.8 / 2.54  # Convert cm to inches
    subplot_size = fig_width_in / 3  # Each subplot should be square
    fig_height_in = fig_width_in / 3  # Keep aspect ratio square

    # Set up the plotting grid
    fig, axes = plt.subplots(1, 3, figsize=(fig_width_in, fig_height_in), sharey=False)
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    # fig.canvas.manager.set_window_title(f"{inspect.stack()[0][3]}")

    # Adjust subplot spacing
    plt.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.2, wspace=0.5)

    # labels for each subplot
    letters = [r"\textbf{(a)}", r"\textbf{(b)}", r"\textbf{(c)}"]
    positions = [(0.09, 0.99), (0.427, 0.99), (0.764, 0.99)]

    for letter, (x_pos, y_pos) in zip(letters, positions):
        fig.text(x_pos, y_pos, letter,
             verticalalignment="top", horizontalalignment="left")

    y_limits = [(0.96, 1.02), (4, 10), (0.02, 0.1)]  # Custom y-axis limits

    # Create a plot for each metric
    for i, (ax, metric, title, ylim) in enumerate(zip(axes, plt_conf["metrics"], plt_conf["titles"], y_limits)):
        # Use seaborn's barplot for each metric
        sns.barplot(
            data=filtered_df,
            x="grid",
            y=metric,
            hue="model_method",
            ax=ax,
            palette=plt_conf["palette"]
        )
        # ax.set_title(title)
        ax.set_xlabel("Grid")
        ax.set_ylabel(plt_conf["titles"][i])
        # ax.legend(title="Method", loc="upper right")
        ax.legend(fontsize=6, markerscale=1, labelspacing=0.05, handletextpad=0.5, loc="upper right")
        ax.set_ylim(ylim)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # ax.set_aspect("equal")  # Forces equal scaling

    # Adjust layout
    # plt.tight_layout()

    # saves plot to external file
    plt.savefig(DIC_PREDICT_BYGRID_PLOT)

    # shows plot
    # plt.show()

def tst_cross_bygrid_plot():
    # Load the data from the CSV file
    df = pd.read_csv(TST_METRICS)

    # load config
    plt_conf = plot_config()

    # Set up the plotting grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 18), sharey=False)
    fig.canvas.manager.set_window_title(f"{inspect.stack()[0][3]}")

    y_limits = [(0.95, 1.02), (0, 2.5), (0, 0.05)]  # Custom y-axis limits

    # Iterate over model methods and metrics to create plots
    model_methods = ["linear", "cubic", "multiquadric"]
    for i, model_method in enumerate(model_methods):
        method_df = df[df['model_method'] == model_method]
        for j, (metric, title, ylim) in enumerate(zip(plt_conf["metrics"], plt_conf["titles"], y_limits)):
            ax = axes[i, j]
            sns.barplot(
                data=method_df,
                x="grid",
                y=metric,
                hue="test_method",
                ax=ax,
                palette=plt_conf["palette"]
            )
            ax.set_title(f"{model_method} model")
            ax.set_xlabel("Grid")
            ax.set_ylabel(metric.upper())
            ax.legend(title="Test method", loc="upper right")
            ax.set_ylim(ylim)
            ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save plot to external file
    plt.savefig(TST_CROSS_BYGRID_PLOT)

    # Show plot
    # plt.show()

def tst_cross_bygrid_r2_plot():
    # Load the data from the CSV file
    df = pd.read_csv(TST_METRICS)

    # load config
    plt_conf = plot_config()

    # Define figure size
    fig_width_in = 13.8 / 2.54  # Convert cm to inches
    subplot_size = fig_width_in / 3  # Each subplot should be square
    fig_height_in = fig_width_in / 3  # Keep aspect ratio square

    # Set up the plotting grid
    fig, axes = plt.subplots(1, 3, figsize=(fig_width_in, fig_height_in), sharey=False)

    # Adjust subplot spacing
    plt.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.2, wspace=0.5)

    # labels for each subplot
    letters = [r"\textbf{(a)}", r"\textbf{(b)}", r"\textbf{(c)}"]
    positions = [(0.09, 0.99), (0.427, 0.99), (0.764, 0.99)]

    for letter, (x_pos, y_pos) in zip(letters, positions):
        fig.text(x_pos, y_pos, letter,
             verticalalignment="top", horizontalalignment="left")

    y_limits = [(0.94, 1.02)]  # Custom y-axis limits

    # Iterate over model methods and metrics to create plots
    model_methods = ["linear", "cubic", "multiquadric"]
    for i, model_method in enumerate(model_methods):
        method_df = df[df['model_method'] == model_method]
        sns.barplot(
            data=method_df,
            x="grid",
            y=plt_conf["metrics"][0],
            hue="test_method",
            ax=axes[i],
            palette=plt_conf["palette"]
        )
        axes[i].set_xlabel("Grid")
        axes[i].set_ylabel(plt_conf["titles"][0])
        axes[i].legend(fontsize=6, markerscale=1, labelspacing=0.05, handletextpad=0.5, loc="upper right")
        axes[i].set_ylim(y_limits[0])
        axes[i].yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Adjust layout
    # plt.tight_layout()

    # Save plot to external file
    plt.savefig(TST_CROSS_BYGRID_R2_PLOT)

    # Show plot
    # plt.show()

def rev_interp_bygrid_plot():
    # Load the data from the CSV file
    df = pd.read_csv(REV_INTERP_METRICS)

    # load config
    plt_conf = plot_config()

    # Define figure size
    fig_width_in = 13.8 / 2.54  # Convert cm to inches
    subplot_size = fig_width_in / 3  # Each subplot should be square
    fig_height_in = fig_width_in / 3  # Keep aspect ratio square

    # Set up the plotting grid
    fig, axes = plt.subplots(1, 3, figsize=(fig_width_in, fig_height_in), sharey=False)
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.canvas.manager.set_window_title(f"{inspect.stack()[0][3]}")

    # Adjust subplot spacing
    plt.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.2, wspace=0.5)

    # labels for each subplot
    letters = [r"\textbf{(a)}", r"\textbf{(b)}", r"\textbf{(c)}"]
    positions = [(0.09, 0.99), (0.427, 0.99), (0.764, 0.99)]

    for letter, (x_pos, y_pos) in zip(letters, positions):
        fig.text(x_pos, y_pos, letter,
             verticalalignment="top", horizontalalignment="left")

    y_limits = [(0, 1.5), (0, 0.001), (0, 0.6)]  # Custom y-axis limits

    # Create a plot for each metric
    for i, (ax, metric, title, ylim) in enumerate(zip(axes, plt_conf["metrics"], plt_conf["titles"], y_limits)):
        sns.barplot(
            data=df,
            x="grid",
            y=metric,
            hue="method",
            ax=ax,
            palette=plt_conf["palette"]
        )
        # ax.set_title(title)
        ax.set_xlabel("Grid")
        ax.set_ylabel(plt_conf["titles"][i])
        # ax.legend(title="Method", loc="upper right")
        ax.legend(fontsize=6, markerscale=1, labelspacing=0.05, handletextpad=0.5, loc="upper right")
        ax.set_ylim(ylim)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # ax.set_aspect("equal")  # Forces equal scaling

    # Adjust layout
    # plt.tight_layout()

    # Save plot to external file
    plt.savefig(REV_INTERP_BYGRID_PLOT)

    # Show plot
    # plt.show()

def time_grid_plot():
    """ Creates a single figure with two subplots: Interpolation Time vs Grid & Training Time vs Grid """
    
    # Load the data from CSV files
    df_interp = pd.read_csv(INTERP_METRICS)
    df_train = pd.read_csv(TRAIN_METRICS)

    # Aggregate data
    aggregated_interp = df_interp.groupby(["grid", "method"], as_index=False)["interpolation_duration"].sum()
    aggregated_train = df_train.groupby(["grid", "method"], as_index=False)["training_duration"].sum()

    # Ensure method column follows this order
    method_order = ["linear", "cubic", "multiquadric"]
    aggregated_interp["method"] = pd.Categorical(aggregated_interp["method"], categories=method_order, ordered=True)
    aggregated_train["method"] = pd.Categorical(aggregated_train["method"], categories=method_order, ordered=True)

    # Load plot configuration
    plt_conf = plot_config()
    
    # Define figure size
    fig_width_in = 13.8 / 2.54  # Convert cm to inches
    subplot_size = fig_width_in / 2  # Each subplot should be square
    fig_height_in = fig_width_in / 2  # Keep aspect ratio square

    # Set up the plotting grid
    fig, axes = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in), sharey=False)

    # Adjust subplot spacing
    plt.subplots_adjust(
        left=0.095,
        right=0.995,
        top=0.88,
        bottom=0.15,
        wspace=0.4
    )

    # labels for each subplot
    letters = [r"\textbf{(a)}", r"\textbf{(b)}"]
    positions = [(0.095, 0.97), (0.62, 0.97)]

    for letter, (x_pos, y_pos) in zip(letters, positions):
        fig.text(x_pos, y_pos, letter,
             verticalalignment="top", horizontalalignment="left")
    
    # custom y limits
    y_limits = [(0.95, 1.04), (0, 0.1), (0, 1)]  # Custom y-axis limits

    # First subplot: Interpolation Time vs Grid
    sns.barplot(
        data=aggregated_interp,
        x="grid",
        y="interpolation_duration",
        hue="method",
        palette=plt_conf["palette"],
        ax=axes[0]  # Use left subplot
    )
    # axes[0].set_title("Grid vs Time")
    axes[0].set_xlabel("Grid")
    axes[0].set_ylabel("Interpolation time [s]")
    axes[0].legend(fontsize=6, markerscale=1, labelspacing=0.05, handletextpad=0.5, loc="upper left")
    axes[0].grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    axes[0].set_ylim(0, 6000)
    axes[0].yaxis.set_major_formatter(FuncFormatter(conditional_comma_formatter))


    # Second subplot: Training Time vs Grid
    sns.barplot(
        data=aggregated_train,
        x="grid",
        y="training_duration",
        hue="method",
        palette=plt_conf["palette"],
        ax=axes[1]  # Use right subplot
    )
    # axes[1].set_title("Grid vs Time")
    axes[1].set_xlabel("Grid")
    axes[1].set_ylabel("Training time [s]")
    axes[1].legend(fontsize=6, markerscale=1, labelspacing=0.05, handletextpad=0.5, loc="upper left")
    axes[1].grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    axes[1].set_ylim(0, 17500)
    axes[1].yaxis.set_major_formatter(FuncFormatter(conditional_comma_formatter))


    # Adjust layout to prevent overlapping
    # plt.tight_layout()

    # Save the combined plot
    plt.savefig(TIME_GRID_PLOT, dpi=300)  # Adjust filename as needed

    # Show the combined figure
    # plt.show()


if __name__ == "__main__":
    tst_simple_bygrid_plot()
    dic_predict_bygrid_plot()
    # tst_cross_bygrid_plot()
    tst_cross_bygrid_r2_plot()
    rev_interp_bygrid_plot()
    time_grid_plot()
