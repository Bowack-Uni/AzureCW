#Imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.runconfig import RunConfiguration
from azureml.core import Workspace, Dataset, Run
from azureml.core.experiment import Experiment

#Initialize workspace and azure compute.
#Requires config.json in directory
ws = Workspace.from_config()
compute_name = "AzureCW-compute"
compute_target = ComputeTarget(workspace=ws, name=compute_name)
run_config = RunConfiguration()
run_config.target = compute_target

#Start experiment
experiment_name = 'random-forest-classification'
experiment = Experiment(workspace=ws, name='experiment_test_name')
run = experiment.start_logging(snapshot_directory=None)

#Load data from Azure
#data = pd.read_csv('processed_data.csv')
dataset = Dataset.get_by_name(ws, name='processed_data', version=3)
data = dataset.to_pandas_dataframe()

#Print data
run.log_table('data_sample', data.head(5).to_dict(orient='list'))

#Remove '# Columns: time','source_file', 'source_folder' columns
data_cols = list(data.columns)
#changed '# Columns: time' to 'time' since csv cols changed
data_cols.remove('time')
data_cols.remove('source_file')
data_cols.remove('source_folder')

#Create labels and features
x = data[data_cols]
y = data['source_folder']
run.log_table('data_sample_cols_removed', x.head(20).to_dict(orient='list'))

#Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
run.log_list('classes', list(le.classes_))

#Split into train test datasets using crossvalidation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Create classifier
randForestModel = RandomForestClassifier(n_estimators=50, random_state=0)

#Cross-validation
crossvalidation_results = cross_val_score(randForestModel, x_train, y_train, cv=3)
run.log('cross_val_mean', crossvalidation_results.mean())
run.log_list('cross_val_scores', crossvalidation_results.tolist())

#Test set performance of final model
randForestModel.fit(x_train, y_train)
y_predictions = randForestModel.predict(X = x_test)
report = classification_report(y_test, y_predictions, 
                            target_names=le.classes_,
                            output_dict=True)

for label, metrics in report.items():
    if isinstance(metrics, dict):
        for metric_name, metric_value in metrics.items():
            run.log(f'{label}_{metric_name}', metric_value)
run.complete()

#Future work: Grid search for hyperparameter tuning plus other models like SVM, Neural Networks, etc.
#Also implement exportation of model for deployment.
#Also implement unit and performance tests.
#Once deployed model can be used in presentation.