import csv
import os
import time
import pandas as pd

# Variables
DATA = r"data/cleaned"
INT_P = r"data/raw/centroids.csv"
X_TRAIN = os.path.join(DATA, "x_train.csv")
X_TEST = os.path.join(DATA, "x_test.csv")
METRICS = r"metrics/interpolation_metrics.csv"
# INFILE = os.path.join(DATA, "x_test.csv")
# NEW_FNAME = os.path.join(DATA, "x_test_forces.csv")
INFILE = os.path.join(DATA, "dic_x_edited.csv")
NEW_FNAME = os.path.join(DATA, "dic_x_forces_2.csv")

# setting global vars
IN_FILES = [X_TRAIN, X_TEST]
GRIDS = [20, 30, 40]
METHODS = ["linear", "cubic", "multiquadric"]
BUFF_TSHOLD = 100

def extractor():
    # imports centroids' parameters of each test (single line) into separate arrays
    try:
        with open(INFILE, mode='r') as file:
            reader = csv.reader(file)
            # next(reader)
            bg_bf = []
            # for each simulation
            for k, row in enumerate(reader, start=1):
                bf = []
                # for each timestep
                for j in range (0,20):
                    # initialize arrays
                    # def_x = []
                    # def_y = []
                    # def_xy = []
                    grid_def_x = []
                    grid_def_y = []
                    grid_def_xy = []
                    c = j*4961*3+j*2
                    x_force = row[c]
                    y_force = row[c+1]

                    bf.append(x_force)
                    bf.append(y_force)

                    for i in range(0, len(grid_def_x)):
                        bf.append(grid_def_x[i])
                        bf.append(grid_def_y[i])
                        bf.append(grid_def_xy[i])

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

    extractor()

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