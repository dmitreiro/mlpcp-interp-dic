"""
Tool to compile csv files.
"""

import glob
import os
import csv
import pandas as pd
import re
import time

# Accessing variables
MYCSVDIR = r"data/raw"
DATA_PROC = r"data/processed"
X = os.path.join(DATA_PROC, "x_compiled.csv")
Y = os.path.join(DATA_PROC, "y_compiled.csv")

# In case of DIC samples
# MYCSVDIR = r"data/dic"
# DATA_PROC = r"data/processed"
# X = os.path.join(DATA_PROC, "dic_x_compiled.csv")
# Y = os.path.join(DATA_PROC, "dic_y_compiled.csv")

# Buffer threshold for dumping intermediate files
X_BUFF_TSHOLD = 100

# Start the timer
start_time = time.time()

# Get all the csv files in that directory (assuming they have the extension .csv)
csvfiles = glob.glob(os.path.join(MYCSVDIR, "*.csv"))
total_files = len(csvfiles)

# Checking for previous data files
if os.path.isfile(X):
    os.remove(X)
if os.path.isfile(Y):
    os.remove(Y)

final_rows = []
final_y = []

for index, cs in enumerate(csvfiles, start=1):
    rows = []
    with open(cs, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            rows.append(row)
    rows = [x for f in rows for x in f]
    final_rows.append(rows)
    final_y.append(re.findall(r"\d+(?:\.\d+)?", cs))

    # Print progress
    print(f"Processed {index}/{total_files} files")

    # If buffer x file has more than X_BUFF_TSHOLD lines, it is dumped
    if len(final_rows) == X_BUFF_TSHOLD:
        # Print progress
        print(f"Dumping x buffer file")
        p = pd.DataFrame(final_rows)
        p.to_csv(X, mode="a", header=False, index=False)
        final_rows = []

print("Dataframe x and y data")
p = pd.DataFrame(final_rows)
pf = pd.DataFrame(final_y, columns=["F", "G", "H", "L", "M", "N", "sigma0", "k", "n"])

print("Writting final x and y data")
p.to_csv(X, mode="a", header=False, index=False)
pf.to_csv(Y)

# End the timer and calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

# Convert elapsed time to minutes and seconds
elapsed_minutes = int(elapsed_time // 60)
elapsed_seconds = int(elapsed_time % 60)

# Print total elapsed time in "minutes:seconds" format
print(
    f"Finished processing {total_files} files in {elapsed_minutes}:{elapsed_seconds:02d} minutes."
)
