import pandas as pd
import os
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error
)
import time
import joblib

# Variables
DATA = r"data/cleaned"
MODELS = r"models"
Y = os.path.join(DATA, "dic_y.csv")
METRICS = r"metrics/dic_prediction_metrics.csv"

GRIDS = [20, 30, 40]
METHODS = ["linear", "cubic", "multiquadric"]

def test_and_evaluate(grid, method, test_method):
    # IMPORT THE FILTERED DATA FOR TESTING
    # THE DATA SHOULD ALREADY BE NORMALIZED
    # THE DATA SHOULD ONLY CONTAIN USEFUL SIMULATIONS (Fxy20>Fxy19)
    # CHANGE NUMBERS ON THE MODEL NAME FILE FOR THE DESIRED MODEL

    # construct paths to the testing files
    x_test = os.path.join(
        DATA, f"dic_x_{grid}_{test_method}.csv"
    )
    xgb_model = os.path.join(
        MODELS, f"xgb_{grid}_{method}.joblib"
    )
    scaler_file = os.path.join(
        MODELS, f"scaler_{grid}_{method}.joblib"
    )

    # load feature and target data
    try:
        print(f"Loading data from {x_test} and {Y}")
        X_test = pd.read_csv(x_test, header=None)
        y_test = pd.read_csv(Y)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # get number of columns and points
    n_cols = len(X_test.columns)
    points = int((n_cols/20-2)/3)

    # define columns for X_test
    l=[]
    for x in range(1,21): # each timestep
        l.append("Force_x_"+str(x))
        l.append("Force_y_"+str(x))
        for p in range(1, points+1):  # elements number
            l.append("Strain_x_"+str(p)+"_"+str(x))
            l.append("Strain_y_"+str(p)+"_"+str(x))
            l.append("Strain_xy_"+str(p)+"_"+str(x))
    
    # assign defined column names to X_train
    X_test.columns = l
    print(f"X_train shape: {X_test.shape}")

    # loading scaler
    try:
        scaler = joblib.load(scaler_file)
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        print(f"Error loading or applying scaler: {e}")
        return

    # load trained model
    try:
        print("Loading xgb model...")
        modelo = joblib.load(xgb_model)
        #print(modelo.get_params())
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # start timer
    start_time = time.time()

    # predict training values
    try:
        y_test_pred = modelo.predict(X_test_scaled)
    except Exception as e:
        print(f"Error predicting values: {e}")
        return
    
    # end the timer and calculate elapsed time in minutes and seconds
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int(elapsed_time % 60)

    # print total elapsed time in "minutes:seconds" format
    print(
        f"Finished testing {xgb_model} in {elapsed_minutes}:{elapsed_seconds:02d} minutes."
    )
    
    # performance on testing data
    try:
        r2_test = r2_score(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
    except Exception as e:
        print(f"Error calculating performance metrics: {e}")
        return
    
    print(f'R-squared on {test_method} test for {grid}_{method} model: {r2_test}')
    print(f'MAE on {test_method} test for {grid}_{method} model: {mae_test}')
    print(f'MAPE on {test_method} test for {grid}_{method} model: {mape_test}')

    # saves predicted y data to csv file
    pred_params_path = os.path.join(DATA, f"dic_y_pred_{grid}_{method}_{test_method}.csv")
    df = pd.DataFrame(y_test_pred)
    df.to_csv(pred_params_path, mode="w", header=True, index=False)

    return {
        "grid": grid,
        "model_method": method,
        "test_method": test_method,
        "r2": r2_test,
        "mae": mae_test,
        "mape": mape_test,
        "testing_duration": elapsed_time
    }

def main():
    """
    Main function to start code execution.
    """

    # start timer
    start_time = time.time()

    # check if the file exists and delete it if it does
    if os.path.exists(METRICS):
        os.remove(METRICS)

    # iterate over the main folder numbers and subfolder numbers to train and evaluate models
    for grid in GRIDS:
        for method in METHODS:
            for test_method in METHODS:
                result = test_and_evaluate(grid, method, test_method)
                if result == None:  # ensure result is not None
                    return 1
                # save results
                result_df = pd.DataFrame([result])
                write_header = not os.path.exists(METRICS)
                result_df.to_csv(METRICS, mode="a", header=write_header, index=False)
                print(f"Testing performance metrics saved to {METRICS}")

    # end the timer and calculate elapsed time in minutes and seconds
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int(elapsed_time % 60)

    # print total elapsed time in "minutes:seconds" format
    print(
        f"Total elapsed time: {elapsed_minutes}:{elapsed_seconds:02d} minutes."
    )

if __name__ == "__main__":
    exit(main())