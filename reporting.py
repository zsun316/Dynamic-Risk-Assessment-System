"""
Author: Zhaohan Sun
Date: July, 2022
This script is used to generate a confusion matrix and PDF report
"""

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions
import seaborn as sns

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

root_path = os.path.dirname(os.path.abspath(__file__))
dataset_csv_path = os.path.join(root_path, config['output_folder_path'])
test_data_path = os.path.join(root_path, config['test_data_path'])
prod_deployment_path = os.path.join(root_path, config['prod_deployment_path'])
cf_path = os.path.join(root_path, config['output_model_path'])

##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    # load model and calculate the predictions
    test_filename = 'testdata.csv'

    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    test_data = pd.read_csv(os.path.join(test_data_path, test_filename), index_col=False)

    y_test = test_data['exited'].values.reshape(-1,1).ravel()
    X_test = test_data.drop(['corporation','exited'], axis=1).values.reshape(-1,3)

    y_pred = model.predict(X_test)

    #Generate the confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                   cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_title('Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Save the visualization of the Confusion Matrix.

    plt.show()
    fig = ax.get_figure()
    fig.savefig(os.path.join(cf_path, 'confusionmatrix.png'))
    return





if __name__ == '__main__':
    score_model()
