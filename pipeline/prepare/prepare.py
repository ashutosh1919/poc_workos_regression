import os
import yaml
import sys
from scipy import stats
import pandas as pd
import pickle

params = yaml.safe_load(open('params.yaml'))['prepare']

if len(sys.argv)!=3:
    print("Not passed enough arguements to prepare stage.")
    sys.exit(1)

print("Starting the prepare stage...")

co_relation_threshold = params['co_relation_threshold']

os.makedirs(os.path.join("data", "prepared"), exist_ok=True)
output_train = os.path.join('data', 'prepared', 'train.pkl')
output_test = os.path.join('data', 'prepared', 'test.pkl')

input_train = sys.argv[1]
input_test = sys.argv[2]

def read_dataset(input_path):
    df = pd.read_pickle(input_path)
    return df

def feature_selection(train_df,test_df):
    temp_train_df = pd.DataFrame()
    temp_test_df = pd.DataFrame()
    ct = 0

    for x in train_df:
        pearson_coef, p_value = stats.pearsonr(train_df[x], train_df[len(train_df.columns) - 1])
        print(f'Pearson coefficient for {x} columns is {pearson_coef}')

        #Feature selection on the bases of pearson_co_relation.
        if abs(pearson_coef) < co_relation_threshold or (len(train_df.columns) - 1 == x):
            temp_train_df[ct] = train_df[x]
            temp_test_df[ct] = test_df[x]
            ct += 1

    print(f'{ct} out of {len(train_df.columns)} features selected.')
    save_as_pkl(temp_train_df, temp_test_df)

def save_as_pkl(train_df,test_df):
    train_df.to_pickle(output_train)
    test_df.to_pickle(output_test)

train_df = read_dataset(input_train)
test_df = read_dataset(input_test)

feature_selection(train_df,test_df)

print('Prepare stage completed successfylly...')