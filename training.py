from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)


root_path = os.path.dirname(os.path.abspath(__file__))
dataset_csv_path = os.path.join(root_path, config['output_folder_path'])
model_path = os.path.join(root_path, config['output_model_path'])


#################Function for training the model
def train_model():
    data_filename = 'finaldata.csv'
    model_filename = 'trainedmodel.pkl'

    #use this logistic regression for training
    logit_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='ovr', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)

    # lgb_model = to be added for better performance
    # Now we just fit the simple logistic regression to our data

    train_data = pd.read_csv(os.path.join(dataset_csv_path, data_filename), index_col=False)
    X_train = train_data.drop(['corporation','exited'], axis=1).values.reshape(-1,3)
    y_train = train_data['exited'].values.reshape(-1,1).ravel()

    model = logit_model.fit(X_train, y_train)

    #write the trained model in a file called trainedmodel.pkl
    with open(os.path.join(model_path, model_filename), 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    train_model()
