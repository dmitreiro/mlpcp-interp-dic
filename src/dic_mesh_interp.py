import numpy as np
import csv
import os
import time
import pandas as pd
from typing import Tuple
from numpy.typing import NDArray
from scipy.interpolate import Rbf, RBFInterpolator

# Variables
DATA = r"data/cleaned"
INT_P = r"data/processed/dic_subset_coords_cent.csv"
X_TRAIN = os.path.join(DATA, "x_train_dic.csv")
X_TEST = os.path.join(DATA, "x_test_dic.csv")
METRICS = r"metrics/interpolation_metrics.csv"

# setting global vars
IN_FILES = [X_TEST]
GRIDS = [30]
METHODS = ["multiquadric"]
BUFF_TSHOLD = 1

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

    points = np.column_stack((x, y))                 # shape (N, 2)
    grid_points = np.column_stack((grid_x, grid_y))  # shape (M, 2)

    # if grid_x == None:
    #     return None

    # imports centroids' parameters of each test (single line) into separate arrays
    try:
        print(f"Starting generating {fname}")
        with open(infile, mode='r') as file:
            reader = csv.reader(file)
            # next(reader) # skips header
            bg_bf = []
            # for each simulation
            for k, row in enumerate(reader, start=1):
                print(f"Processing row {k}...")
                # checks if number of elements matches with coordinates given
                if (len(row)/20-2)/3 != len(x):
                    raise ValueError(f"Error: Number of elements in row {k} does not match number of coordinates ({len(x)}).")
                
                bf = []
                # for each timestep
                for j in range (0,20):
                    print(f"  Timestep {j+1}/20")
                    # initialize arrays
                    def_x = []
                    def_y = []
                    def_xy = []
                    grid_def_x = []
                    grid_def_y = []
                    grid_def_xy = []
                    c = j*len(x)*3+j*2
                    x_force = row[c]
                    y_force = row[c+1]

                    # for each element
                    print("    Extracting element data...")
                    for i in range(c+2, c+2+len(x)*3, 3):
                        def_x.append(row[i])
                        def_y.append(row[i + 1])
                        def_xy.append(row[i + 2])

                    print("    Converting lists to numpy arrays and stacking...")
                    def_x = np.array(def_x, dtype=float)
                    def_y = np.array(def_y, dtype=float)
                    def_xy = np.array(def_xy, dtype=float)
                    values = np.column_stack((def_x, def_y, def_xy))

                    # print("x coord:", x.min(), x.max())
                    # print("y coord:", y.min(), y.max())
                    # print("points.shape:", points.shape)

                    # print("def_x range:", def_x.min(), def_x.max())
                    # print("def_y range:", def_y.min(), def_y.max())
                    # print("def_xy range:", def_xy.min(), def_xy.max())
                    # print("values.shape:", values.shape)

                    print("    Creating RBF interpolator...")
                    rbf = RBFInterpolator(points, values, kernel=method, epsilon=1)

                    print("    Interpolating on the grid...")
                    grid_def = rbf(grid_points)

                    print("    Removing NaN values and appending forces and strains...")
                    grid_def = np.nan_to_num(grid_def)
                    bf.append(x_force)
                    bf.append(y_force)
                    bf.extend(grid_def.ravel())

                    # print("bf.shape:", np.array(bf).shape)

                print("Appending data to buffer...")
                # dump buffer to big buffer
                bg_bf.append(bf)

                if len(bg_bf) == BUFF_TSHOLD:
                    print(f"Writing {len(bg_bf)} rows to {new_fname}...")
                    write_header = not os.path.isfile(new_fname)

                    with open(new_fname, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)

                        if write_header:
                            n_cols = len(bg_bf[0])
                            writer.writerow(range(n_cols))  # 0,1,2,...,N-1

                        writer.writerows(bg_bf)

                    bg_bf.clear()
        
        with open(new_fname, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(bg_bf)

    except Exception as e:
        print(f"Error interpolating input file: {e}")
        return None

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
    x, y = [], []
    try:
        with open(INT_P, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                x.append(row[0])  # first column
                y.append(row[1])  # second column

        # convert to float array
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

    except Exception as e:
        print(f"Error importing centroid coordinates: {e}")
        return 1
    
    print(f"Imported {len(x)} coordinate points")

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