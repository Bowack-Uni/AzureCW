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
    #data.to_csv("processed_data.csv", index=False)
    print("Merged and saved to processed_data.csv")
else:
    print("No valid data")

##Errors exsist in firwst itteration CSV which must be corrected
#Ensure columns are named correctly
data.columns = ['time', 'avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 
              'avg_rss23', 'var_rss23', 'source_folder', 'source_file']

#find rows where time has spaces. There is an error in the csv
#where all the values end up in 'time' for a few instances
brokenRows = data[~data['time'].astype(str).contains(' ')]
normalRows = data[data['time'].astype(str).contains(' ')]

#Fix the broken rows
fixedRows = []
for index, row in brokenRows.iterrows():
    #Split the time into the correct values
    values = row['time'].split()
    #Reassignes the values
    fixedRow = {
    'time': float(values[0]),          
            'avg_rss12': float(values[1]),                
            'var_rss12': float(values[2]),    
            'avg_rss13': float(values[3]),
            'var_rss13': float(values[4]),
            'avg_rss23': float(values[5]),
            'var_rss23': float(values[6]),
            'source_folder': row['source_folder'],  
            'source_file': row['source_file']      
    }
    #This may break continutity of data when added back into the 
    #original csv but since my models are not time series this should not matter
    fixedRows.append(fixedRow)

#Combine the normal and fixed rows to recreate the original dataset
data = pd.concat([normalRows, pd.DataFrame(fixedRows)], ignore_index=True)
#Send to CSV
data.to_csv("processed_data.csv", index=False)