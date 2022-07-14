"""
Author: Zhaohan Sun
Date: July, 2022
This script is used to conduct diagnosis
"""
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess
import ast
import pickle



##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

root_path = os.path.dirname(os.path.abspath(__file__))
dataset_csv_path = os.path.join(root_path, config['output_folder_path'])
test_data_path = os.path.join(root_path, config['test_data_path'])
prod_deployment_path = os.path.join(root_path, config['prod_deployment_path'])
model_path = os.path.join(root_path, config['output_model_path'])


##################Function to get model predictions
def model_predictions():

    test_filename = 'testdata.csv'

    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    test_data = pd.read_csv(os.path.join(test_data_path, test_filename), index_col=False)

    y_test = test_data['exited'].values.reshape(-1,1).ravel()
    X_test = test_data.drop(['corporation','exited'], axis=1).values.reshape(-1,3)

    y_pred = model.predict(X_test)

    return y_pred
    #return return value should be a list containing all predictions

##################Function to get summary statistics

def dataframe_summary():

    test_filename = 'testdata.csv'
    test_data = pd.read_csv(os.path.join(test_data_path, test_filename), index_col=False)

    df_covar = test_data.drop(['corporation', 'exited'], axis=1)

    mp = {}
    for col in df_covar.columns:
        mp[col] = {'mean': np.round(df_covar[col].mean(),3),
                   'std': np.round(df_covar[col].std(),3),
                   'median': np.round(df_covar[col].median(),3)}
    return mp

##################Function to check missingness
def missingness_summary():

    test_filename = 'testdata.csv'
    test_data = pd.read_csv(os.path.join(test_data_path, test_filename), index_col=False)

    mp_miss_ratio, num_rows = [], test_data.shape[0]
    for col in test_data.columns:
        mp_miss_ratio.append(test_data[col].isna().sum()/num_rows)

    return mp_miss_ratio
    # return #return value should be a list containing all summary statistics

##################Function to get timings

def ingestion_time():
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'ingestion.py'], capture_output=True)
    run_time = timeit.default_timer() - starttime
    return run_time


def training_time():
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'training.py'], capture_output=True)
    run_time = timeit.default_timer() - starttime
    return run_time


def execution_time():
    #calculate timing of training.py and ingestion.py
    ingestion_times, training_times = [], []
    for _ in range(25):
        time = ingestion_time()
        ingestion_times.append(time)
        time = training_time()
        training_times.append(time)

    res = [
        {'ingestion_time': np.round(np.mean(ingestion_times),3)},
        {'training_time': np.round(np.mean(training_times),3)}
    ]

    return res


##################Function to check dependencies
def outdated_packages_list():
    # get a list of out dated python packages.
    dependencies = subprocess.run(
        ['pip-outdated', 'requirements.txt'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8')

    dep = dependencies.stdout
    dep = dep.translate(str.maketrans('', '', ' \t\r'))
    dep = dep.split('\n')
    dep = [dep[3]] + dep[5:-3]
    dep = [s.split('|')[1:-1] for s in dep]
    col_names = dep[0]
    dep = pd.DataFrame(dep[1:], columns=col_names)
    return dep


if __name__ == '__main__':
    step_pred = model_predictions()
    print(f'The model predictions are {step_pred}')

    step_summary = dataframe_summary()
    for key, val in step_summary.items():
        print(f'The summary statistics of column {key} is {val}')

    step_missing = missingness_summary()
    print(f'The ratio of missingness of each column in test data is {step_missing}')

    step_recordtime = execution_time()
    print(f'The excution times for ingestion and training are {step_recordtime}')

    step_checkpack = outdated_packages_list()
    print(f'The result of checking out dated packages are')
    print(pd.DataFrame(step_checkpack))
