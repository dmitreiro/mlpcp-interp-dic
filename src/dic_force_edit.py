import numpy as np
import csv
import os
import time
import pandas as pd
from typing import Tuple
from numpy.typing import NDArray
from scipy.interpolate import Rbf

# Variables
DATA = r"data/cleaned"
INT_P = r"data/raw/centroids.csv"
X_TRAIN = os.path.join(DATA, "x_train.csv")
X_TEST = os.path.join(DATA, "x_test.csv")
METRICS = r"metrics/interpolation_metrics.csv"
# INFILE = os.path.join(DATA, "x_test.csv")
# NEW_FNAME = os.path.join(DATA, "x_test_forces.csv")
INFILE = os.path.join(DATA, "dic_x.csv")
NEW_FNAME = os.path.join(DATA, "dic_x_edited.csv")

# setting global vars
IN_FILES = [X_TRAIN, X_TEST]
GRIDS = [20, 30, 40]
METHODS = ["linear", "cubic", "multiquadric"]
BUFF_TSHOLD = 100

# Extracted forces (hardcoded) to replace the ones in dic_x.csv
FORCES = [1135.5819396972656,1098.9316730499268,
          1306.5255756378174,1262.3196601867676,
          1416.1174926757812,1367.3380336761477,
          1496.455047607422,1444.3808822631836,
          1560.3042526245115,1505.6238136291504,
          1612.575340270996,1555.744197845459,
          1656.3741092681885,1597.7279624938965,
          1693.6850624084473,1633.4689197540283,
          1725.800937652588,1664.2209014892578,
          1753.6707038879397,1690.8771057128906,
          1778.009105682373,1714.1331481933594,
          1799.332748413086,1734.4854583740234,
          1818.063133239746,1752.33736038208,
          1834.529521942139,1768.0064430236816,
          1849.0001220703125,1781.7478866577148,
          1861.6945991516116,1793.7767486572266,
          1872.79833984375,1804.269306182861,
          1882.4673957824707,1813.3762130737305,
          1890.835391998291,1821.2240753173828,
          1898.012580871582,1827.9251251220703]

def force_replacer():
    # imports centroids' parameters of each test (single line) into separate arrays
    try:
        with open(INFILE, mode='r') as file:
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
                    # grid_def_x = []
                    # grid_def_y = []
                    # grid_def_xy = []
                    c = j*4961*3+j*2
                    # x_force = row[c]
                    # y_force = row[c+1]

                    # for each element
                    for i in range(c+2, c+2+4961*3, 3):
                        def_x.append(row[i])
                        def_y.append(row[i + 1])
                        def_xy.append(row[i + 2])

                    def_x = np.array(def_x, dtype=float)
                    def_y = np.array(def_y, dtype=float)
                    def_xy = np.array(def_xy, dtype=float)

                    bf.append(FORCES[j*2])
                    bf.append(FORCES[j*2+1])

                    for i in range(0, len(def_x)):
                        bf.append(def_x[i])
                        bf.append(def_y[i])
                        bf.append(def_xy[i])

                # dump buffer to big buffer
                bg_bf.append(bf)
                
                # dump big buffer to file
                if len(bg_bf) == BUFF_TSHOLD and not os.path.isfile(NEW_FNAME):
                    p = pd.DataFrame(bg_bf)
                    p.to_csv(NEW_FNAME, mode="a", header=True, index=False)
                    bg_bf = []
                elif len(bg_bf) == BUFF_TSHOLD and os.path.isfile(NEW_FNAME):
                    p = pd.DataFrame(bg_bf)
                    p.to_csv(NEW_FNAME, mode="a", header=False, index=False)
                    bg_bf = []
        
        p = pd.DataFrame(bg_bf)
        p.to_csv(NEW_FNAME, mode="a", header=False, index=False)

    except Exception as e:
        print(f"Error fetching input file: {e}")
        return None

def main():
    """
    Main function to start code execution.
    """

    # start timer
    start_time = time.time()

    force_replacer()

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