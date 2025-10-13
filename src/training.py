import pandas as pd
import time
import joblib
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os

# Variables
DATA = r"data/cleaned"
MODELS = r"models"
Y_TRAIN = os.path.join(DATA, "y_train.csv")
METRICS = r"metrics/training_performance_metrics.csv"

GRIDS = [20, 30, 40]
METHODS = ["linear", "cubic", "multiquadric"]

# Function to train and evaluate model
def train_and_evaluate(grid: int, method: str):
    """
    Function to train and evaluate model, based on `grid` and `method` input variables.
    """

    # construct paths to the training files
    x_train = os.path.join(
        DATA, f"x_train_{grid}_{method}.csv"
    )

    # load feature and target data
    try:
        print(f"Loading data from {x_train} and {Y_TRAIN}")
        X_train = pd.read_csv(x_train)
        y_train = pd.read_csv(Y_TRAIN)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # get number of columns and points
    n_cols = len(X_train.columns)
    points = int((n_cols/20-2)/3)

    # define columns for X_train
    l = []
    for x in range(1, 21): # each timestep
        l.append("Force_x_" + str(x))
        l.append("Force_y_" + str(x))
        for p in range(1, points+1):  # elements number
            l.append("Strain_x_" + str(p) + "_" + str(x))
            l.append("Strain_y_" + str(p) + "_" + str(x))
            l.append("Strain_xy_" + str(p) + "_" + str(x))

    # assign the defined column names to X_train
    X_train.columns = l
    # print(f"X_train shape: {X_train.shape}")

    # scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # set path for scaler file
    scaler_file = os.path.join(
        MODELS, f"scaler_{grid}_{method}.joblib"
    )

    # dump the scaler to a file
    joblib.dump(scaler, scaler_file)

    # start timer for training
    start_time_training = time.monotonic()

    # train the model on the training data
    try:
        print(f"Starting train...")
        modelo = MultiOutputRegressor(xgb.XGBRegressor(learning_rate=0.02, max_depth=4, n_estimators=1000, tree_method="hist", device="cpu")).fit(X_train_scaled, y_train)
    except Exception as e:
        print(f"Error training model: {e}")
        return

    # stop timer for training
    end_time_training = time.monotonic()
    training_duration = end_time_training - start_time_training
    print(
        f"Training duration for model {grid}_{method}: {training_duration} seconds"
    )

    # save the trained model
    model_filename = os.path.join(
        MODELS, f"xgb_{grid}_{method}.joblib")
    try:
        joblib.dump(modelo, model_filename)
        print(f"Model saved as {model_filename}")
    except Exception as e:
        print(f"Error saving model: {e}")
        return

    # predict on training data
    try:
        y_train_pred = modelo.predict(X_train_scaled)
    except Exception as e:
        print(f"Error predicting on training data: {e}")
        return

    # performance on training data
    try:
        r2_train = r2_score(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    except Exception as e:
        print(f"Error calculating performance metrics: {e}")
        return

    print(f"R-squared on Train Data for {grid}_{method}: {r2_train}")
    print(f"MAE on Train Data for {grid}_{method}: {mae_train}")
    print(f"MAPE on Train Data for {grid}_{method}: {mape_train}")

    return {
        "grid": grid,
        "method": method,
        "r2": r2_train,
        "mae": mae_train,
        "mape": mape_train,
        "training_duration": training_duration
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

    # iterate over the main folder numbers and subfolder numbers to train and evaluate models
    for grid in GRIDS:
        for method in METHODS:
            result = train_and_evaluate(grid, method)
            if result == None:  # ensure result is not None
                return 1
            # save results
            result_df = pd.DataFrame([result])
            write_header = not os.path.exists(METRICS)
            result_df.to_csv(METRICS, mode="a", header=write_header, index=False)
            print(f"Training performance metrics saved to {METRICS}")

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