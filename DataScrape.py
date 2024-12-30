import pandas as pd
import os

data = []
directory = r"C:\Users\bowes\Downloads\activity+recognition+system+based+on+multisensor+data+fusion+arem"

#Itterate through folders
for directory, folders, files in os.walk(directory):
    for file in files:
        if file.endswith(".csv"):
            csvdata = (pd.read_csv(os.path.join(directory, file), skiprows=4))
            csvdata['source_folder'] = os.path.basename(directory)
            csvdata['source_file'] = file
            data.append(csvdata)

#Merge dataframes
data = pd.concat(data, ignore_index=True)
data.to_csv("processed_data.csv", index=False)