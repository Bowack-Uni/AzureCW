#Imports
from random import shuffle
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.runconfig import RunConfiguration
from azureml.core import Workspace, Dataset, Run, Model
from azureml.core.experiment import Experiment
import unittest
import time
import joblib
import mlflow
import mlflow.sklearn


#Initialize workspace and azure compute.
#Requires config.json in directory
ws = Workspace.from_config()
compute_name = "AzureCW-compute"
compute_target = ComputeTarget(workspace=ws, name=compute_name)
run_config = RunConfiguration()
run_config.target = compute_target

#Start experiment
#experiment_name = 'random-forest-classification'
#experiment = Experiment(workspace=ws, name='experiment_test_name')
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment("experiment_test_name")
#run = experiment.start_logging(snapshot_directory=None)

#Load data from Azure
#data = pd.read_csv('processed_data.csv')
dataset = Dataset.get_by_name(ws, name='processed_data', version=3)
data = dataset.to_pandas_dataframe()

#Print data
#run.log_table('data_sample', data.head(5).to_dict(orient='list'))

#Remove '# Columns: time','source_file', 'source_folder' columns
data_cols = list(data.columns)
#changed '# Columns: time' to 'time' since csv cols changed
data_cols.remove('time')
data_cols.remove('source_file')
data_cols.remove('source_folder')

#Create labels and features
x = data[data_cols]
y = data['source_folder']
#run.log_table('data_sample_cols_removed', x.head(20).to_dict(orient='list'))

#Scale features & shuffle data
#scaler = StandardScaler()
#x = scaler.fit_transform(x)
#x = shuffle(x)

#Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
#run.log_list('classes', list(le.classes_))

#Split into train test datasets using crossvalidation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)

#Defines models and parameters. Format was taken from Claude AI
models = {
    'rf': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [10, None]
        }
    },
    'svm': {
        'model': SVC(),
        'params': {
            'C': [1, 10],
            'kernel': ['rbf', 'linear']
        }
    },
    'nn': {
        'model': MLPClassifier(),
        'params': {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu', 'tanh']
        }
    }
}

#Train and evaluate models. New training and evaluation code was improved by Claude AI
#to incorporate grid search & multible models
#best_score = 0
#best_model = None
#best_model_name = None
#for name, model_info in models.items():
#    start_time = time.time()
#    grid_search = GridSearchCV(
#        model_info['model'],
#        model_info['params'],
#        cv=3
#    )
#    grid_search.fit(x_train, y_train)
    #mlflow.log_param('model_name', best_model_name)
    #mlflow.log_metric('best_score', best_score)
    #mlflow.log_params(grid_search.best_params_)
    #mlflow.log_metric('training_time', time.time() - start_time)

    #run.log(f'{name}_best_score', grid_search.best_score_)
    #run.log(f'{name}_best_params', grid_search.best_params_)
    #run.log(f'{name}_training_time', time.time() - start_time)
    
#    if grid_search.best_score_ > best_score:
#        best_score = grid_search.best_score_
#        best_model = grid_search.best_estimator_
#        best_model_name = name

#Test set performance of best model
randForestModel = RandomForestClassifier(n_estimators=50, random_state=0)
randForestModel.fit(x_train, y_train)

y_predictions = randForestModel.predict(X = x_test)
report = classification_report(y_test, y_predictions, 
                            target_names=le.classes_,
                            output_dict=True)

with mlflow.start_run():

    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, metric_value in metrics.items():
                #run.log(f'{label}_{metric_name}', metric_value)
                mlflow.log_metric(f'{label}_{metric_name}', metric_value)

#Export model and preprocessing objects
#lflow.sklearn.log_model(best_model, artifact_path="models", registered_model_name=best_model_name)
    input_example = x_test.iloc[[0]]
    #input_example = x_test.head(1)
    #print(input_example)

    mlflow.sklearn.log_model(randForestModel, artifact_path="models", registered_model_name="randForestModel", input_example=input_example)

    #joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    #mlflow.log_artifact('scaler.pkl')
    mlflow.log_artifact('label_encoder.pkl')

#joblib.dump(scaler, 'scaler.joblib')
#joblib.dump(le, 'label_encoder.joblib')

#joblib.dump(best_model, f'{best_model_name}_model.joblib')
#joblib.dump(randForestModel, 'randForestModel.joblib')

#Model.register(workspace=ws, model_path='scaler.joblib', model_name='scaler')
#Model.register(workspace=ws, model_path='label_encoder.joblib', model_name='label_encoder')

#Model.register(workspace=ws, model_path=f'{best_model_name}_model.joblib', model_name=best_model_name)
#Model.register(workspace=ws, model_path='randForestModel.joblib', model_name='randForestModel')

#Basic unit tests. [Implement more for invalid data, etc.]
#def test_model_prediction():
#    assert best_model.predict(x_test[:1]).shape == (1,)

#def test_data_scaling():
#    assert np.allclose(x.mean(axis=0), 0, atol=1e-8)
#    assert np.allclose(x.std(axis=0), 1, atol=1e-8)

#test_model_prediction()
#test_data_scaling()

#run.complete()

#Create classifier
#randForestModel = RandomForestClassifier(n_estimators=50, random_state=0)

#Cross-validation
#crossvalidation_results = cross_val_score(randForestModel, x_train, y_train, cv=3)
#run.log('cross_val_mean', crossvalidation_results.mean())
#run.log_list('cross_val_scores', crossvalidation_results.tolist())

#Test set performance of final model
#randForestModel.fit(x_train, y_train)
#y_predictions = randForestModel.predict(X = x_test)
#report = classification_report(y_test, y_predictions, 
#                            target_names=le.classes_,
#                            output_dict=True)

#for label, metrics in report.items():
#    if isinstance(metrics, dict):
#        for metric_name, metric_value in metrics.items():
#            run.log(f'{label}_{metric_name}', metric_value)
#run.complete()

#Future work: Grid search for hyperparameter tuning plus other models like SVM, Neural Networks, etc.
#Also implement exportation of model for deployment.
#Also implement unit and performance tests.
#Once deployed model can be used in presentation.