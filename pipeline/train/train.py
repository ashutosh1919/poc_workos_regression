import pandas as pd
import json
import yaml
import pickle
import os
import sys
from sklearn.linear_model import LinearRegression


params = yaml.safe_load(open('params.yaml'))['train']

if len(sys.argv)!=2:
    print("Not passed enough arguements to train stage.")
    sys.exit(1)

print("Starting the training stage...")

fit_intercept = params['fit_intercept']
normalize = params['normalize']
n_jobs = params['n_jobs']
copy_X = params['copy_X']

os.makedirs(os.path.join("model", "Regression_checkpoints"), exist_ok=True)
output_model = os.path.join("model", "Regression_checkpoints", "best.pkl")

input_train = sys.argv[1]

def load_data(pkl_filepath):
    df = pd.read_pickle(pkl_filepath)
    return df

def save_model(model):
    with open(output_model, 'wb') as file:
        pickle.dump(model, file)

def linear_regr_model(train_df):
    # Create Linear Regression Object
    lm2 = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs)
    Y2 = train_df.pop(len(train_df.columns) - 1)
    X2 = train_df

    # Fit (Train) the model
    lm2.fit(X2, Y2)

    print("Intercept for the model is", lm2.intercept_, "and the scope is", lm2.coef_)

    # Save model
    save_model(lm2)

train_df = load_data(input_train)
linear_regr_model(train_df)

print("Training stage completed...")