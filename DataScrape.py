#Import libraries
import pandas as pd
import os

#Initialize variables
data = []
directory = r'C:\Users\bowes\Downloads\ars'

#Itterate through folders
for dirpath, folders, files in os.walk(directory):
    for file in files:
        if file.endswith(".csv"):
            path = os.path.join(dirpath, file)
            print(f"Processing file: {path}")
            try:
                #Skip error producing lines
                csvdata = pd.read_csv(path, skiprows=4, on_bad_lines='skip')
                csvdata['source_folder'] = os.path.basename(dirpath)
                csvdata['source_file'] = file
                data.append(csvdata)
            except Exception as e:
                print(f"Error processing file {path}: {e}")

#Merge dataframes
if data:
    data = pd.concat(data, ignore_index=True)
    data.to_csv("processed_data.csv", index=False)
    print("Merged and saved to processed_data.csv")
else:
    print("No valid data")