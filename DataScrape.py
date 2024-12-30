import pandas as pd
import os

data = []
directory = r"C:\Users\bowes\Downloads\ars"

#Itterate through folders
for directory, folders, files in os.walk(directory):
    for file in files:
        if file.endswith(".csv"):
            path = os.path.join(directory, file)
            print(path)
            csvdata = (pd.read_csv(path, skiprows=4))
            csvdata['source_folder'] = os.path.basename(directory)
            csvdata['source_file'] = file
            data.append(csvdata)

#Merge dataframes
data = pd.concat(data, ignore_index=True)
data.to_csv("processed_data.csv", index=False)