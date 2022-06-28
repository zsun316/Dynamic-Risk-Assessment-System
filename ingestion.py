import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f)

root_path = os.path.dirname(os.path.abspath(__file__))
input_folder_path = os.path.join(root_path, config['input_folder_path'])
output_folder_path = os.path.join(root_path, config['output_folder_path'])


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    files = os.listdir(input_folder_path)

    # Get only the csv files, and merge these csv files.
    data_files = []
    df_final = pd.DataFrame(columns=['corporation','lastmonth_activity',
                                     'lastyear_activity','number_of_employees',
                                     'exited'])
    for file in files:
        if file.split('.')[-1] == 'csv':
            df_cur = pd.read_csv(os.path.join(input_folder_path, file), index_col=False)
            df_final = pd.concat([df_final, df_cur]).drop_duplicates().reset_index(drop=True)

    print(df_final)
    # save the merged fies.
    df_final.to_csv(os.path.join(output_folder_path, 'finaldata.csv'),
                    index=False)


if __name__ == '__main__':
    merge_multiple_dataframe()
