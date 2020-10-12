# This script downloads the dataset locally, loads only a chunk of the dataset for the training.

import os
import yaml
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open('params.yaml'))['extract']

test_samples = params['test_samples']
train_samples = params['train_samples']

os.makedirs(os.path.join("data", "raw"), exist_ok=True)
output_train = os.path.join('data', 'raw', 'train.pkl')
output_test = os.path.join('data', 'raw', 'test.pkl')

def fetch_datasets(train_samples=100, test_samples=20):

    df = pd.DataFrame(load_boston().data)
    df[len(df.columns)] = load_boston().target
    train_df, test_df = train_test_split(df, test_size= test_samples/(test_samples + train_samples))
    return train_df, test_df

def count_elements(df, df_type):
    ct = 0
    for x in df[0]:
        ct += 1
    print(f'{df_type} data containes {ct} data points.')

def save_datasets(train_df, test_df):
    train_df.to_pickle(output_train)
    test_df.to_pickle(output_test)
    count_elements(train_df, "Training")
    count_elements(test_df, "Testing")

train_df, test_df = fetch_datasets(train_samples, test_samples)
save_datasets(train_df, test_df)