import os
import pandas as pd
import matplotlib.pyplot as plt

PROC_DIR = r"data/processed"
COORDS = os.path.join(PROC_DIR, "dic_subset_coords.csv")
COORDS_CENTERED = os.path.join(PROC_DIR, "dic_subset_coords_cent.csv")

# load data
df = pd.read_csv(COORDS, header=None)
X = df.iloc[:, 0]
Y = df.iloc[:, 1]

# global bounds
x_min, x_max = X.min(), X.max()
y_min, y_max = Y.min(), Y.max()
width_x  = x_max - x_min
height_y = y_max - y_min

print("X min, max: ", x_min, x_max)
print("Y min, max: ", y_min, y_max)
print("Original X range: ", x_min, " to ", x_max)
print("Original Y range: ", y_min, " to ", y_max)

TARGET = 30.0

margin_x = (TARGET - width_x) / 2
margin_y = (TARGET - height_y) / 2

# shift coordinates
df.iloc[:, 0] = X - x_min + margin_x
df.iloc[:, 1] = Y - y_min + margin_y

# plot to verify
plt.figure()
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], s=1)
plt.gca().set_aspect("equal", adjustable="box")
plt.xlim(0, TARGET)
plt.ylim(0, TARGET)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Correctly centered geometry in [0, 30] × [0, 30]")
plt.show()

# save to csv file
df.to_csv(COORDS_CENTERED, index=False, header=False)
