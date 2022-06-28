from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

root_path = os.path.dirname(os.path.abspath(__file__))
dataset_csv_path = os.path.join(root_path, config['output_folder_path'])
test_data_path = os.path.join(root_path, config['test_data_path'])
output_model_path = os.path.join(root_path, config['output_model_path'])


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data,
    #and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    score_filename = "latestscore.txt"
    model_filename = 'trainedmodel.pkl'
    test_filename = 'testdata.csv'


    with open(os.path.join(output_model_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    test_data = pd.read_csv(os.path.join(test_data_path, test_filename), index_col=False)

    y_test = test_data['exited'].values.reshape(-1,1).ravel()
    X_test = test_data.drop(['corporation','exited'], axis=1).values.reshape(-1,3)

    y_pred = model.predict(X_test)
    f1scores = f1_score(y_pred, y_test)
    print(f1scores)
    
    with open(os.path.join(output_model_path, score_filename), 'wb') as f:
        f.write(f1scores)


if __name__ == '__main__':
    score_model()
