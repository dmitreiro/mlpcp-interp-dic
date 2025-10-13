import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Paths
CURVES_PLOT = r"res/dic_params_curves_2.pdf"

# Reload CSV files
stress_strain_df = pd.read_csv(r"data/synthetic_dic/stress_strain.csv")
stress_rolling_dir_df = pd.read_csv(r"data/synthetic_dic/stress_rolling_dir.csv")
r_rolling_dir_df = pd.read_csv(r"data/synthetic_dic/r_rolling_dir.csv")

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.facecolor": (1,1,1),
    "figure.facecolor": (1,1,1),
    # "font.family": "serif",
    "font.family": "Palatino",
    "font.size": 7,
    "legend.fontsize": 6,
    "legend.edgecolor": "black"
})

# Define the color palette
palette = sns.color_palette("tab10")[:3]

# Define figure size
fig_width_in = 15 / 2.54  # Convert cm to inches
fig_height_in = fig_width_in * (10/18)  # Keep aspect ratio square

# Create the 2x3 subplots figure with specified color palette
fig, axs = plt.subplots(2, 3, figsize=(fig_width_in, fig_height_in))

# Adjust subplot spacing
plt.subplots_adjust(
    left=0.07,
    right=0.99,
    top=0.95,
    bottom=0.11,
    wspace=0.45,
    hspace=0.55
)

# Add subplot labels
letters = [r"\textbf{(a)}", r"\textbf{(b)}", r"\textbf{(c)}",
           r"\textbf{(d)}", r"\textbf{(e)}", r"\textbf{(f)}"]
positions = [
        (0.072, 0.99), (0.413, 0.99), (0.755, 0.99),
        (0.072, 0.48), (0.413, 0.48), (0.755, 0.48)
    ]

for letter, (x_pos, y_pos) in zip(letters, positions):
        fig.text(x_pos, y_pos, letter,
             verticalalignment="top", horizontalalignment="left")

# Plot (1,1) - stress_strain
axs[0, 0].plot(stress_strain_df['x_axis'], stress_strain_df['y_test'], label='y_test', color=palette[0], linewidth=1)
axs[0, 0].plot(stress_strain_df['x_axis'], stress_strain_df['original_model'], label='original_model', color=palette[1], linewidth=1)
axs[0, 0].plot(stress_strain_df['x_axis'], stress_strain_df['DIC'], label='DIC', color=palette[2], linewidth=1)
# axs[0, 0].set_title("Stress-Strain")
axs[0, 0].set_xlabel("Equivalent plastic strain [-]")
axs[0, 0].set_ylabel("Yield stress [MPa]")
axs[0, 0].set_xticks([0, 0.1, 0.2, 0.3])
axs[0, 0].set_xlim(0, 0.3)
axs[0, 0].set_yticks([0, 100, 200, 300, 400, 500, 600])
axs[0, 0].set_ylim(0, 600)
axs[0, 0].legend(markerscale=1, labelspacing=0.05, handletextpad=0.5, loc="upper left", bbox_to_anchor=(0.35, 0.38))

# Plot (2,1) - stress_strain_error
axs[1, 0].plot(stress_strain_df['x_axis'], stress_strain_df['original_model_error'], label='original_model', color=palette[1], linewidth=1)
axs[1, 0].plot(stress_strain_df['x_axis'], stress_strain_df['DIC_error'], label='DIC', color=palette[2], linewidth=1)
axs[1, 0].axhline(0, color='black', linewidth=0.8, linestyle='--')
# axs[1, 0].set_title("Stress-Strain Error")
axs[1, 0].set_xlabel("Equivalent plastic strain [-]")
axs[1, 0].set_ylabel("Rel. Error [\%]")
axs[1, 0].set_xticks([0, 0.1, 0.2, 0.3])
axs[1, 0].set_xlim(0, 0.3)
axs[1, 0].set_yticks([-4, -3, -2, -1, 0, 1, 2])
axs[1, 0].set_ylim(-4, 2)
axs[1, 0].legend(markerscale=1, labelspacing=0.05, handletextpad=0.5, loc="upper left")

# Plot (1,2) - stress_rolling_dir
axs[0, 1].plot(stress_rolling_dir_df['x_axis'], stress_rolling_dir_df['y_test'], label='y_test', color=palette[0], linewidth=1)
axs[0, 1].plot(stress_rolling_dir_df['x_axis'], stress_rolling_dir_df['original_model'], label='original_model', color=palette[1], linewidth=1)
axs[0, 1].plot(stress_rolling_dir_df['x_axis'], stress_rolling_dir_df['DIC'], label='DIC', color=palette[2], linewidth=1)
# axs[0, 1].set_title("Stress vs Rolling Direction")
axs[0, 1].set_xlabel("Angle w.r.t. rolling direction [$^\circ$]")
axs[0, 1].set_ylabel("Initial yield stress [MPa]")
axs[0, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
axs[0, 1].set_xlim(0, 90)
axs[0, 1].set_yticks([120, 130, 140, 150, 160, 170, 180])
axs[0, 1].set_ylim(120, 180)
axs[0, 1].legend(markerscale=1, labelspacing=0.05, handletextpad=0.5, loc="upper left", bbox_to_anchor=(0.2, 1.0))

# Plot (2,2) - stress_rolling_dir_error
axs[1, 1].plot(stress_rolling_dir_df['x_axis'], stress_rolling_dir_df['original_model_error'], label='original_model', color=palette[1], linewidth=1)
axs[1, 1].plot(stress_rolling_dir_df['x_axis'], stress_rolling_dir_df['DIC_error'], label='DIC', color=palette[2], linewidth=1)
axs[1, 1].axhline(0, color='black', linewidth=0.8, linestyle='--')
# axs[1, 1].set_title("Stress vs Rolling Direction Error")
axs[1, 1].set_xlabel("Angle w.r.t. rolling direction [$^\circ$]")
axs[1, 1].set_ylabel("Rel. Error [\%]")
axs[1, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
axs[1, 1].set_xlim(0, 90)
axs[1, 1].set_yticks([-6, -4, -2, 0, 2, 4])
axs[1, 1].set_ylim(-6, 4)
axs[1, 1].legend(markerscale=1, labelspacing=0.05, handletextpad=0.5, loc="upper left")

# Plot (1,3) - r_rolling_dir
axs[0, 2].plot(r_rolling_dir_df['x_axis'], r_rolling_dir_df['y_test'], label='y_test', color=palette[0], linewidth=1)
axs[0, 2].plot(r_rolling_dir_df['x_axis'], r_rolling_dir_df['original_model'], label='original_model', color=palette[1], linewidth=1)
axs[0, 2].plot(r_rolling_dir_df['x_axis'], r_rolling_dir_df['DIC'], label='DIC', color=palette[2], linewidth=1)
# axs[0, 2].set_title("r-value vs Rolling Direction")
axs[0, 2].set_xlabel("Angle w.r.t. rolling direction [$^\circ$]")
axs[0, 2].set_ylabel("r-value [-]")
axs[0, 2].set_xticks([0, 15, 30, 45, 60, 75, 90])
axs[0, 2].set_xlim(0, 90)
axs[0, 2].set_yticks([0, 1, 2, 3, 4, 5, 6])
axs[0, 2].set_ylim(0, 6)
axs[0, 2].legend(markerscale=1, labelspacing=0.05, handletextpad=0.5, loc="upper left", bbox_to_anchor=(0, 0.38))

# Plot (2,3) - r_rolling_dir_error
axs[1, 2].plot(r_rolling_dir_df['x_axis'], r_rolling_dir_df['original_model_error'], label='original_model', color=palette[1], linewidth=1)
axs[1, 2].plot(r_rolling_dir_df['x_axis'], r_rolling_dir_df['DIC_error'], label='DIC', color=palette[2], linewidth=1)
axs[1, 2].axhline(0, color='black', linewidth=0.8, linestyle='--')
# axs[1, 2].set_title("r-value vs Rolling Direction Error")
axs[1, 2].set_xlabel("Angle w.r.t. rolling direction [$^\circ$]")
axs[1, 2].set_ylabel("Rel. Error [\%]")
axs[1, 2].set_xticks([0, 15, 30, 45, 60, 75, 90])
axs[1, 2].set_xlim(0, 90)
axs[1, 2].set_yticks([-8, -6, -4, -2, 0, 2, 4, 6])
axs[1, 2].set_ylim(-8, 6)
axs[1, 2].legend(markerscale=1, labelspacing=0.05, handletextpad=0.5, loc="upper left", bbox_to_anchor=(0.35, 0.57))

for row in axs:
    for ax in row:
        # ax.grid(True, which='both', axis='both')
        ax.set_xlim(left=0)


# plt.tight_layout()
plt.savefig(CURVES_PLOT, format="pdf")
# plt.show()
