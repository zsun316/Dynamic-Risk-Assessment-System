import requests
import subprocess
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"
with open('config.json','r') as f:
    config = json.load(f)

root_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(root_path, config['output_model_path'])


#Call each API endpoint and store the responses
#response1 = subprocess.run(['curl','127.0.0.1:8000/prediction?filename=testdata.csv'], capture_output=True).stdout
response1 = requests.get('http://127.0.0.1:8000/prediction?filename=testdata.csv').text

#response2 = subprocess.run(['curl','127.0.0.1:8000/scoring?filename=testdata.csv'], capture_output=True).stdout
response2 = requests.get('http://127.0.0.1:8000/scoring?filename=testdata.csv').text

#response3 = subprocess.run(['curl','127.0.0.1:8000/summarystats?filename=testdata.csv'], capture_output=True).stdout
response3 = requests.get('http://127.0.0.1:8000/summarystats?filename=testdata.csv').text

# response4 = subprocess.run(['curl','127.0.0.1:8000/diagnostics?filename=testdata.csv'], capture_output=True).stdout
response4 = requests.get('http://127.0.0.1:8000/diagnostics?filename=testdata.csv').text

#combine all API responses
#responses = #combine reponses here


#write the responses to your workspace
with open(os.path.join(model_path, 'apireturns.txt'), 'w') as file:
    file.write('----------- Summary statistics for Ingested Data -----------\n\n')
    file.write(str(response3))
    file.write('\n\n ----------- Diagnostics Summary ----------- \n\n')
    file.write(str(response4))
    file.write('\n\n ----------- Model Prediction ----------- \n\n')
    file.write(str(response1))
    file.write('\n\n ----------- Model Score ----------- \n\n')
    file.write(str(response2))
