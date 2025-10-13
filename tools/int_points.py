"""
Tool to calculate integration points for C3D8R elements using nodes.
"""

import os
import csv
import time

# Variables
MYCSVDIR = r"data/raw"
DATA_PROC = r"data/processed"
EL = os.path.join(MYCSVDIR, "elements.csv")
ND = os.path.join(MYCSVDIR, "nodes.csv")
INT_P = os.path.join(DATA_PROC, "int_points.csv")

# Start the timer
start_time = time.time()

# Checking for previous data files
if os.path.isfile(INT_P):
    os.remove(INT_P)

with open(EL, 'r') as f_elem, open(ND, 'r') as f_nodes, open(INT_P, 'w') as f_intp:
    f_elem_r = csv.reader(f_elem)
    f_nodes_r = csv.reader(f_nodes)
    f_intp_w = csv.writer(f_intp)
    coords = []
    n_row = next(f_nodes_r)

    for element in f_elem_r:
        el_coords = []
        el_x = []
        el_y = []
        el_z = []
        for node in element:
            el_x.append(float(n_row[(int(node)-1)*3]))
            el_y.append(float(n_row[(int(node)-1)*3+1]))
            el_z.append(float(n_row[(int(node)-1)*3+2]))

        x_val = sum(el_x)/len(el_x)
        y_val = sum(el_y)/len(el_y)
        z_val = sum(el_z)/len(el_z)
        el_coords.extend([x_val, y_val, z_val])
        coords.append(el_coords)
    
    f_intp_w.writerows(coords)


# End the timer and calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

# Convert elapsed time to minutes and seconds
elapsed_minutes = int(elapsed_time // 60)
elapsed_seconds = int(elapsed_time % 60)

# Print total elapsed time in "minutes:seconds" format
print(f"Finished in {elapsed_minutes}:{elapsed_seconds:02d} minutes.")
