"""
Author: Zhaohan Sun
Date: July, 2022
This script is used to conduct full machine learning training/retraining process
"""

import training
import scoring
import deployment
import diagnostics
import ingestion
import reporting
import os
import json
import ast
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

root_path = os.path.dirname(os.path.abspath(__file__))
input_folder_path = os.path.join(root_path, config['input_folder_path'])
dataset_csv_path = os.path.join(root_path, config['output_folder_path'])
test_data_path = os.path.join(root_path, config['test_data_path'])
prod_deployment_path = os.path.join(root_path, config['prod_deployment_path'])
model_path = os.path.join(root_path, config['output_model_path'])

def main():
    # file names:
    record_filename = 'ingestedfiles.txt'
    data_filename = 'finaldata.csv'
    test_filename = 'testdata.csv'

    ##################Check and read new data
    #first, read ingestedfiles.txt
    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    with open(os.path.join(prod_deployment_path, record_filename), 'r') as f:
        str_list = f.read()
        prev_files = ast.literal_eval(str_list)
    # print(prev_files)

    new_files = os.listdir(input_folder_path)
    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    if not set(new_files) - set(prev_files):
        print('No new data ingested, process ended.')
        return None

    df_prev = pd.read_csv(os.path.join(dataset_csv_path, data_filename), index_col=False)
    ingestion.merge_multiple_dataframe(prev_files, df_prev)
    print('new data ingested')

    ##################Deciding whether to proceed, part 2
    # if you found new score is greater than previous score, than no model drift,
    # should stop, otherwise do retraining
    with open(os.path.join(prod_deployment_path, "latestscore.txt")) as f:
        prev_score = ast.literal_eval(f.read())

    df_new = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'), index_col=False)
    y_new = df_new['exited'].values.reshape(-1,1).ravel()
    X_new = df_new.drop(['corporation','exited'], axis=1).values.reshape(-1,3)

    test_data = pd.read_csv(os.path.join(test_data_path, test_filename), index_col=False)

    y_test = test_data['exited'].values.reshape(-1,1).ravel()
    X_test = test_data.drop(['corporation','exited'], axis=1).values.reshape(-1,3)

    logit_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='ovr', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)

    model_new = logit_model.fit(X_new, y_new)
    y_new_pred = model_new.predict(X_test)

    new_score = f1_score(y_new_pred, y_test)
    print(new_score)

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score
    #from the model that uses the newest ingested data

    if(new_score >= prev_score):
        print("No model drift occurred")
        return None
    print('Model drift occurred!')

    ##################Re-training
    training.train_model()

    ##################Re-scoring
    scoring.score_model()

    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    deployment.store_model_into_pickle()

    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    reporting.score_model()

    return



if __name__ == '__main__':
    main()
