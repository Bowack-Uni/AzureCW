#Debugging code made with chatgpt. Only used for debugging purposes and to upload datasets to the cloud
from azureml.core import Workspace, Dataset, Datastore
import os

# Connect to your Azure ML workspace
try:
    ws = Workspace.from_config()
    print(f"Successfully connected to workspace: {ws.name}")
except Exception as e:
    print("Error connecting to the workspace:", e)
    exit()

# Define dataset name and file path
dataset_name = "processed_data"
file_path = os.path.join(os.getcwd(), "processed_data.csv")

# Get the default datastore
datastore = ws.get_default_datastore()
print(f"Using datastore: {datastore.name}")

# Upload the file to the datastore
try:
    target_path = "datasets/"  # Path inside the datastore
    datastore.upload_files(
        files=[file_path],
        target_path=target_path,
        overwrite=True,
        show_progress=True,
    )
    print(f"File '{file_path}' uploaded to datastore at path '{target_path}'.")
except Exception as e:
    print("Error uploading file to datastore:", e)
    exit()

# Register the dataset
try:
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, target_path + "processed_data.csv"))
    dataset = dataset.register(
        workspace=ws,
        name=dataset_name,
        description="Processed data for classification",
        tags={"source": "datastore", "type": "csv"},
        create_new_version=True,
    )
    print(f"Dataset '{dataset_name}' registered successfully.")
except Exception as e:
    print(f"Error registering dataset '{dataset_name}':", e)
    exit()

# Test retrieval of the dataset
try:
    dataset = Dataset.get_by_name(ws, name=dataset_name)
    print(f"Dataset '{dataset_name}' retrieved successfully.")
    data = dataset.to_pandas_dataframe()
    print(data.head())
except Exception as e:
    print(f"Error retrieving dataset '{dataset_name}':", e)
