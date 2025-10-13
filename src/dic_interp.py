import numpy as np
import csv
import os
import time
import pandas as pd
from typing import Tuple
from numpy.typing import NDArray
from scipy.interpolate import Rbf
from scipy.stats import qmc

# Variables
DATA = r"data/cleaned"
X = os.path.join(DATA, "dic_x.csv")
METRICS = r"metrics/dic_interpolation_metrics.csv"

# setting global vars
IN_FILES = [X]
GRIDS = [20, 30, 40]
METHODS = ["linear", "cubic", "multiquadric"]
BUFF_TSHOLD = 100

def mesh_gen(n_points: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Defines mesh grid of [`n_points`x`n_points`] inside a 30x30 square
    and filters it to fit cruciform geometry domain, returning a tuple with two
    (`n`,) arrays (`x` and `y` coordinates) of `n` filtered points.
    """

    # define the grid
    try:
        x = np.linspace(0, 30, n_points)
        y = np.linspace(0, 30, n_points)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.flatten(), yy.flatten()])
    except Exception as e:
        print(f"Error generating mesh grid: {e}")
        return None, None

    # conditions inside the square region
    in_main_square = (points[:, 0] >= 0) & (points[:, 0] <= 30) & (points[:, 1] >= 0) & (points[:, 1] <= 30)
    out_excluded_square = ~((points[:, 0] > 15) & (points[:, 0] <= 30) & (points[:, 1] > 15) & (points[:, 1] <= 30))
    out_excluded_circle = ((points[:, 0] - 15)**2 + (points[:, 1] - 15)**2) >= 7**2
    in_fillet_circ_1 = ((points[:, 0] - 12.5)**2 + (points[:, 1] - 24.17)**2) <= 2.5**2
    in_fillet_circ_2 = ((points[:, 0] - 24.17)**2 + (points[:, 1] - 12.5)**2) <= 2.5**2
    out_square_1 = (points[:, 0] > 13.16) & (points[:, 0] < 15) & (points[:, 1] > 21.75) & (points[:, 1] < 24.17)
    out_square_2 = (points[:, 0] > 21.75) & (points[:, 0] < 24.17) & (points[:, 1] > 13.16) & (points[:, 1] < 15)
    square_1_cond = out_square_1 & in_fillet_circ_1
    square_2_cond = out_square_2 & in_fillet_circ_2

    # combine all conditions
    final_region = (
        in_main_square
        & out_excluded_square
        & out_excluded_circle
        & ~out_square_1
        & ~out_square_2
    )
    final_region |= square_1_cond | square_2_cond

    # extract the valid points and separate into x and y coordinates
    valid_points = points[final_region]
    x_coords = valid_points[:, 0]  # all rows, first column
    y_coords = valid_points[:, 1]  # all rows, second column
    
    return x_coords, y_coords

def lhs_points_in_cruciform(n_points: int, seed: int = None):
    if seed is not None:
        np.random.seed(seed)

    # Create Latin Hypercube Sampler
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    samples = sampler.random(n=5 * n_points)  # generate more than needed

    # Scale to domain [0, 30] x [0, 30]
    samples = qmc.scale(samples, [0, 0], [30, 30])

    # Filter based on cruciform domain
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

def interpolator(infile: str, grid: int, method: str, x: NDArray[np.float64], y: NDArray[np.float64]):
    """
    Interpolates `infile` csv data file with a mesh grid of `grid`x`grid` points
    using any `method` from `scipy.interpolate Rbf`. Integration points coordinates `x` and `y` must
    be given.
    """

    # start timer
    start_time = time.time()

    # extract the base name (without extension) from the original file
    bname = os.path.basename(infile)
    bname = os.path.splitext(bname)[0]
    fname = f"{bname}_{grid}_{method}.csv"
    new_fname = os.path.join(DATA, fname)

    # checking for previous data files
    if os.path.isfile(new_fname):
        os.remove(new_fname)

    grid_x, grid_y = mesh_gen(grid)

    # if grid_x == None:
    #     return None

    # imports centroids' parameters of each test (single line) into separate arrays
    # try:
    with open(infile, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        bg_bf = []
        # for each simulation
        for k, row in enumerate(reader, start=1):
            bf = []
            # for each timestep
            for j in range (0,20):
                # initialize arrays
                def_x = []
                def_y = []
                def_xy = []
                grid_def_x = []
                grid_def_y = []
                grid_def_xy = []
                c = j*5000*3+j*2
                x_force = row[c]
                y_force = row[c+1]

                # for each element
                for i in range(c+2, c+2+5000*3, 3):
                    def_x.append(row[i])
                    def_y.append(row[i + 1])
                    def_xy.append(row[i + 2])

                def_x = np.array(def_x, dtype=float)
                def_y = np.array(def_y, dtype=float)
                def_xy = np.array(def_xy, dtype=float)

                # create RBF interpolators for each parameter
                rbf_def_x = Rbf(x, y, def_x, function=method)
                rbf_def_y = Rbf(x, y, def_y, function=method)
                rbf_def_xy = Rbf(x, y, def_xy, function=method)

                # interpolate on the grid
                grid_def_x = rbf_def_x(grid_x, grid_y)
                grid_def_y = rbf_def_y(grid_x, grid_y)
                grid_def_xy = rbf_def_xy(grid_x, grid_y)
                
                # replace nan values with 0
                grid_def_x = np.nan_to_num(grid_def_x)
                grid_def_y = np.nan_to_num(grid_def_y)
                grid_def_xy = np.nan_to_num(grid_def_xy)

                bf.append(x_force)
                bf.append(y_force)

                for i in range(0, len(grid_def_x)):
                    bf.append(grid_def_x[i])
                    bf.append(grid_def_y[i])
                    bf.append(grid_def_xy[i])

            # dump buffer to big buffer
            bg_bf.append(bf)
            
            # dump big buffer to file
            if len(bg_bf) == BUFF_TSHOLD and not os.path.isfile(new_fname):
                p = pd.DataFrame(bg_bf)
                p.to_csv(new_fname, mode="a", header=True, index=False)
                bg_bf = []
            elif len(bg_bf) == BUFF_TSHOLD and os.path.isfile(new_fname):
                p = pd.DataFrame(bg_bf)
                p.to_csv(new_fname, mode="a", header=False, index=False)
                bg_bf = []
    
    p = pd.DataFrame(bg_bf)
    p.to_csv(new_fname, mode="a", header=False, index=False)

    # except Exception as e:
    #     print(f"Error interpolating input file: {e}")
    #     return None

    # end the timer and calculate elapsed time in minutes and seconds
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int(elapsed_time % 60)

    # print total elapsed time in "minutes:seconds" format
    print(
        f"Finished processing {fname} in {elapsed_minutes}:{elapsed_seconds:02d} minutes."
    )

    return {
            "grid": grid,
            "method": method,
            "file": os.path.splitext(os.path.basename(infile))[0],
            "interpolation_duration": elapsed_time
            }

def main():
    """
    Main function to start code execution.
    """

    # start timer
    start_time = time.time()

    # checking for previous data files
    if os.path.exists(METRICS):
        os.remove(METRICS)

    # importing x,y centroids coordinates into arrays
    # x, y = [], []
    # try:
    #     with open(INT_P, 'r') as file:
    #         reader = csv.reader(file)
    #         for row in reader:
    #             x.append(row[0])  # first column
    #             y.append(row[1])  # second column

    #     # convert to float array
    #     x = np.array(x, dtype=float)
    #     y = np.array(y, dtype=float)

    # except Exception as e:
    #     print(f"Error importing centroid coordinates: {e}")
    #     return 1
    
    x, y = lhs_points_in_cruciform(5000, seed=42)

    for grid in GRIDS:
        for method in METHODS:
            for file in IN_FILES:
                result = interpolator(file, grid, method, x, y)
                if result == None:  # ensure result is not None
                    return 1
                # save results
                result_df = pd.DataFrame([result])
                write_header = not os.path.exists(METRICS)
                result_df.to_csv(METRICS, mode="a", header=write_header, index=False)
                print(f"Metrics saved to {METRICS}")

    # end the timer and calculate elapsed time in minutes and seconds
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int(elapsed_time % 60)

    # print total elapsed time in "minutes:seconds" format
    print(
        f"Total elapsed time: {elapsed_minutes}:{elapsed_seconds:02d} minutes."
    )

    return 0

if __name__ == "__main__":
    exit(main())