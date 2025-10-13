import numpy as np
import matplotlib.pyplot as plt

DIC_CENTROIDS = r"res/dic_speckle_pattern.pdf"

# Use your preferred LaTeX and plotting style
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.facecolor": (1,1,1),
    "figure.facecolor": (1,1,1),
    "font.family": "Palatino",
    "font.size": 8,
    "legend.fontsize": 6,
    "legend.edgecolor": "black"
})

# Define figure size to match LaTeX inclusion
fig_height_cm = 6.9
fig_height_in = fig_height_cm / 2.54  # convert to inches
fig_width_in = fig_height_in  # square plot
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))


# Axis labels, ticks, limits
x_min, x_max = 0, 30
y_min, y_max = 0, 30
tick_interval = 5
x_ticks = np.arange(x_min, x_max + tick_interval, tick_interval)
y_ticks = np.arange(y_min, y_max + tick_interval, tick_interval)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_aspect("equal")

# Adjust subplot spacing
plt.subplots_adjust(
    # left=0.03,
    # right=1.03,
    top=0.98,
    bottom=0.14,
    # wspace=0.35
)

# Generate 5000-point speckle pattern using Latin Hypercube
from scipy.stats import qmc

def lhs_points_in_cruciform(n_points: int, seed: int = None):
    if seed is not None:
        np.random.seed(seed)

    sampler = qmc.LatinHypercube(d=2, seed=seed)
    samples = sampler.random(n=5 * n_points)
    samples = qmc.scale(samples, [0, 0], [30, 30])

    valid_points = []
    for x, y in samples:
        in_main_square = (0 <= x <= 30) and (0 <= y <= 30)
        out_excluded_square = not (15 < x <= 30 and 15 < y <= 30)
        out_excluded_circle = ((x - 15)**2 + (y - 15)**2) >= 7**2
        in_fillet_circ_1 = ((x - 12.5)**2 + (y - 24.17)**2) <= 2.5**2
        in_fillet_circ_2 = ((x - 24.17)**2 + (y - 12.5)**2) <= 2.5**2
        out_square_1 = (13.16 < x < 15) and (21.75 < y < 24.17)
        out_square_2 = (21.75 < x < 24.17) and (13.16 < y < 15)

        square_1_cond = out_square_1 and in_fillet_circ_1
        square_2_cond = out_square_2 and in_fillet_circ_2

        final_cond = (
            in_main_square and
            out_excluded_square and
            out_excluded_circle and
            not out_square_1 and
            not out_square_2
        ) or square_1_cond or square_2_cond

        if final_cond:
            valid_points.append((x, y))
            if len(valid_points) == n_points:
                break

    valid_points = np.array(valid_points)
    return valid_points[:, 0], valid_points[:, 1]

# Generate points
x, y = lhs_points_in_cruciform(5000, seed=42)

# Plot speckle pattern
ax.scatter(x, y, c='blue', marker='.', s=1)

# Save and/or show
plt.savefig(DIC_CENTROIDS, bbox_inches='tight')
# plt.show()
