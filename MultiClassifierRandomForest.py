#Imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#Load data
data = pd.read_csv('processed_data.csv')
print(data.head())

#Remove '# Columns: time','source_file', 'source_folder' columns
data_cols = data.columns.tolist()
data_cols.remove('# Columns: time')
data_cols.remove('source_file')
data_cols.remove('source_folder')
print(data[data_cols].head())

#Create labels and features
x = data[data_cols]
y = data['source_folder']

#Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

#Split into train test datasets using crossvalidation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Create classifier
randForestModel = RandomForestClassifier(n_estimators=50, random_state=0)

#Cross-validation
crossvalidation_results = cross_val_score(randForestModel, x_train, y_train, cv=3)
print(crossvalidation_results)
print(crossvalidation_results.mean())

#Test set performance of final model
randForestModel.fit(x_train, y_train)
y_predictions = randForestModel.predict(X = x_test)
print(classification_report(y_test, y_predictions, 
                            target_names=le.classes_))

#Future work: Grid search for hyperparameter tuning plus other models like SVM, Neural Networks, etc.