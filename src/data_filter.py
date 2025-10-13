import pandas as pd
import numpy as np
import random
import os

# Variables
DATA = r"data/cleaned"
X_CRUCIFORM = os.path.join(DATA, "x_cruciform.csv")
Y_CRUCIFORM = os.path.join(DATA, "y_cruciform.csv")
X_TRAIN = os.path.join(DATA, "x_train.csv")
Y_TRAIN = os.path.join(DATA, "y_train.csv")
X_TEST = os.path.join(DATA, "x_test.csv")
Y_TEST = os.path.join(DATA, "y_test.csv")

def main():
    """
    Main function to start code execution.
    """

    try:
        # import X file
        X = pd.read_csv(X_CRUCIFORM, header=None)

        # removes duplicated lines
        X.drop_duplicates(inplace=True)

        # removes lines with NULL
        X.dropna(inplace=True)

        # removes column 1
        # X = X.iloc[:, 1:]

        # import Y file
        y = pd.read_csv(Y_CRUCIFORM, sep=",")

        # column filter
        colunas_selecionadas = [coluna for coluna in y.columns if not coluna.startswith('Unnamed')]
        colunas_sem_nan = [coluna for coluna in colunas_selecionadas if not y[coluna].isnull().all()]
        y = y[colunas_sem_nan]
        # y=y.loc[X.index]

        # get number of columns and points
        n_cols = len(X.columns)
        points = int((n_cols/20-2)/3)

        # put columns into X
        l=[]
        for x in range(1,21):
            l.append("Force_x_"+str(x))
            l.append("Force_y_"+str(x))
            for p in range(1, points+1): # number of elements
                l.append("Strain_x_"+str(p)+"_"+str(x))
                l.append("Strain_y_"+str(p)+"_"+str(x))
                l.append("Strain_xy_"+str(p)+"_"+str(x))
        X.columns = l

        X=X.reset_index(drop=True)
        y=y.reset_index(drop=True)

        # filter to ignore tests in which the force decreased from timestep 19 to 20
        index1=X[(X["Force_y_20"]-X["Force_y_19"]>0) & (X["Force_x_20"]-X["Force_x_19"]>0)].index
        index1

        # apply filter to original dataframe to extract good simulations
        X=X.iloc[index1]
        y=y.iloc[index1]

        # replace the first row in X with a numeric header from 1 to number of columns
        X.columns = list(range(X.shape[1]))

        # set a random seed for reproducibility
        random_state = 42
        np.random.seed(random_state)

        # calculate number of rows to delete (to get a "round number")
        rows_to_delete = len(X) - 2260

        # randomly choose the indices to delete
        indices_to_delete = np.random.choice(X.index, size=rows_to_delete, replace=False)

        # drop the selected indices from both X and y
        X_reduced = X.drop(indices_to_delete)
        y_reduced = y.drop(indices_to_delete)

        X = X_reduced
        y = y_reduced

        X=X.reset_index(drop=True)
        y=y.reset_index(drop=True)

        # separate data for train and test
        r = random.sample(range(0, len(X)), 260) # define number of simulations to test
        X_test = X.loc[r]
        y_test = y.loc[r]
        X_train = X.drop(r)
        y_train = y.drop(r)

    except Exception as e:
        print(f"Error filtering x and y cruciform data: {e}")
        return 1

    # save data to csv
    try:
        X_train.to_csv(X_TRAIN, index=False)
        y_train.to_csv(Y_TRAIN, index=False)
        X_test.to_csv(X_TEST, index=False)
        y_test.to_csv(Y_TEST, index=False)
    except Exception as e:
        print(f"Error saving x and y cruciform data: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())