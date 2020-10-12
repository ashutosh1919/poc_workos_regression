import pandas as pd
import yaml
import pickle
import os
import json
import sys
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

params = yaml.safe_load(open('params.yaml'))['evaluate']

if len(sys.argv)!=3:
    print("Not passed enough arguements to evaluate stage.")
    sys.exit(1)

print("Starting evaluation stage...")

metrics = params['metrics']

data_path = sys.argv[1]
model_ckpt_dir = sys.argv[2]

os.makedirs(os.path.join('.', 'results'), exist_ok=True)
out_file = os.path.join("results", "metrics.json")

out_file_rmse= os.path.join("results", "train_stats_rmse.json")
out_file_rsquare = os.path.join("results", "train_stats_rsquare.json")
out_file_mse = os.path.join("results", "train_stats_mse.json")
out_file_mae = os.path.join("results", "train_stats_mae.json")


def load_data(pkl_filepath):
    df = pd.read_pickle(pkl_filepath)
    return df

def load_model(filepath):
    with open(filepath, 'rb') as file:
        regr_model = pickle.load(file)
    return regr_model

def save_results(schema, out_file):
    text = json.dumps(schema)
    with open(out_file, "w") as f:
        f.write(text)

def model_evalute(test_df, regr_model, metrics):
    Y1 = test_df.pop(len(test_df.columns) - 1)
    X1 = test_df

    Yout1 = regr_model.predict(X1)

    # Error
    for i in metrics:
        if i == "mse" :
            # L2 Loss (mse)
            mse = mean_squared_error(Y1, Yout1)
            print("Mean square error for simple linear regression is", mse)
            save_results({"mse":mse},out_file_mse)

        elif i == "mae":
            # l1 Loss (Mae)
            mae = mean_absolute_error(Y1, Yout1)
            print("Mean absolute error value for simple linear regression is", mae)
            save_results({"mae":mae}, out_file_mae)

        elif i == "rsquare":
            # Rsquare
            rsquare = r2_score(Y1, Yout1)
            print("R-square error value for simple linear regression is", rsquare)
            save_results({"rsquare":rsquare}, out_file_rsquare)

        elif i == "rmse":
            # RMSE
            rmse = mean_squared_error(Y1, Yout1, squared=True)
            print("Root Mean square error value for simple linear regression is", rmse)
            save_results({"rmse":rmse}, out_file_rmse)

# Storing the results is pending.

test_df = load_data(data_path)
regr_model = load_model(model_ckpt_dir)
model_evalute(test_df,regr_model,metrics)


print("Evaluation stage completed...")