from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
import diagnostics
# import predict_exited_from_saved_model
import json
import os
from sklearn.metrics import f1_score


with open('config.json','r') as f:
    config = json.load(f)

root_path = os.path.dirname(os.path.abspath(__file__))
dataset_csv_path = os.path.join(root_path, config['output_folder_path'])
test_data_path = os.path.join(root_path, config['test_data_path'])
prod_deployment_path = os.path.join(root_path, config['prod_deployment_path'])
cf_path = os.path.join(root_path, config['output_model_path'])

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'


# prediction_model = None

@app.route('/')
def index():
    return "Welcome to Dynamic Risk Assessment System"

#######################Prediction Endpoint
@app.route("/prediction", methods=['GET', 'POST'])
def predict():
    model_filename = 'trainedmodel.pkl'

    #call the prediction function you created in Step 3
    if request.method == 'GET':
        filename = request.args.get('filename')
    else:
        filename = request.form('filename')
    df = pd.read_csv(os.path.join(test_data_path, filename), index_col=False)

    y_data = df['exited'].values.reshape(-1,1).ravel()
    X_data = df.drop(['corporation','exited'], axis=1).values.reshape(-1,3)

    # load model
    with open(os.path.join(prod_deployment_path, model_filename), 'rb') as f:
        model = pickle.load(f)

    # prediction
    y_pred = model.predict(X_data)
    return str(y_pred)


#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scorestats():
    model_filename = 'trainedmodel.pkl'

    if request.method == 'GET':
        filename = request.args.get('filename')
    else:
        filename = request.form('filename')

    with open(os.path.join(prod_deployment_path, model_filename), 'rb') as f:
        model = pickle.load(f)

    test_data = pd.read_csv(os.path.join(test_data_path, filename), index_col=False)

    y_test = test_data['exited'].values.reshape(-1,1).ravel()
    X_test = test_data.drop(['corporation','exited'], axis=1).values.reshape(-1,3)

    y_pred = model.predict(X_test)
    f1scores = f1_score(y_pred, y_test)
    return str(round(f1scores,3)) #add return value (a single F1 score number)


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():
    if request.method == 'GET':
        filename = request.args.get('filename')
    else:
        filename = request.form('filename')

    test_data = pd.read_csv(os.path.join(test_data_path, filename), index_col=False)

    df_covar = test_data.drop(['corporation', 'exited'], axis=1)

    mp = {}
    for col in df_covar.columns:
        mp[col] = {'mean': np.round(df_covar[col].mean(),3),
                   'std': np.round(df_covar[col].std(),3),
                   'median': np.round(df_covar[col].median(),3)}
    return mp #return a list of all calculated summary statistics


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagstats():
    if request.method == 'GET':
        filename = request.args.get('filename')
    else:
        filename = request.form('filename')

    step_missing = diagnostics.missingness_summary(filename)
    # print(f'The ratio of missingness of each column in test data is {step_missing}')

    step_recordtime = diagnostics.execution_time()
    # print(f'The excution times for ingestion and training are {step_recordtime}')

    step_checkpack = pd.DataFrame(diagnostics.outdated_packages_list())

    return {'step_missing':step_missing,
            'step_recordtime': step_recordtime,
            'step_checkpack': step_checkpack.T.to_dict('r')}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
