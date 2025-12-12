import os
import csv

DIR = r"data/raw"
PROC_DIR = r"data/processed"
ZERO_STEP = os.path.join(DIR, "Static_0000_0_Numerical_0_0.synthetic.tif.csv")
COORD_OUTPUT = os.path.join(PROC_DIR, "dic_subset_coords.csv")

coords = []

with open(ZERO_STEP, newline='', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=';')
    next(reader, None)  # skip header

    for row in reader:
        # convert decimal comma to float:
        x = float(row[2].replace(",", "."))
        y = float(row[3].replace(",", "."))
        z = float(row[4].replace(",", "."))
        coords.append([x, y, z])

with open(COORD_OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(coords)