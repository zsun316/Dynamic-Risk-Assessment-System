from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f)

root_path = os.path.dirname(os.path.abspath(__file__))
dataset_csv_path = os.path.join(root_path, config['output_folder_path'])
prod_deployment_path = os.path.join(root_path, config['prod_deployment_path'])
model_path = os.path.join(root_path, config['output_model_path'])

####################function for deployment
def store_model_into_pickle():
    # copy the latest pickle file,
    # the latestscore.txt value,
    # and the ingestfiles.txt file into the deployment directory
    record_filename = 'ingestedfiles.txt'
    model_filename = 'trainedmodel.pkl'
    score_filename = "latestscore.txt"


    shutil.copy(os.path.join(model_path, model_filename),
                os.path.join(prod_deployment_path, model_filename))
    shutil.copy(os.path.join(model_path, score_filename),
                os.path.join(prod_deployment_path, score_filename))
    shutil.copy(os.path.join(dataset_csv_path, record_filename),
                os.path.join(prod_deployment_path, record_filename))


if __name__ == '__main__':
    store_model_into_pickle()
