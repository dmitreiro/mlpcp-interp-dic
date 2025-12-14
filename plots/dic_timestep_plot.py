"""
Plots strains and coordinates for a timestep, from a MatchID file.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

DATA = r"data/cleaned"
FILE = os.path.join(DATA, "Static_0000_0_Numerical_20_0.synthetic.tif.csv")
df = pd.read_csv(FILE, sep=';')

cols = [2, 3, 16, 17, 18]
for c in cols:
    df.iloc[:, c] = (
        df.iloc[:, c]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

x = df.iloc[:, 2]
y = df.iloc[:, 3]

for strain, label in [(16, "Strain XX"), (17, "Strain YY"), (18, "Strain XY")]:
    plt.figure()
    plt.scatter(x, y, c=df.iloc[:, strain], s=1, rasterized=True)
    plt.colorbar(label=label)
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.title(label)
    plt.show()