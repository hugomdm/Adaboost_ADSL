
#------ lib packages 
import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

#----------- external imports 
import Adaboost
import data_processing
import models_evaluation

#from sklearn.datasets import load_breast_cancer
#data = load_breast_cancer()
#X = data.data
#y = data.target

arg = sys.argv[1:]
parser = argparse.ArgumentParser(description="Parse command line arguments.")
parser.add_argument("-d", "--data", type=str, required=True,
                    help="Test for binary or multiclass data ['binary', 'multiclass']")

args = parser.parse_args(arg)

X, y = data_processing.split_data(args.data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

if args.data == 'binary': 
    ada = Adaboost.BinaryClassAdaboost(50)

if args.data == 'multiclass': 
    ada = Adaboost.MultiClassAdaboost(50)
    
print(X_train)

ada.fit(X_train, np.array(y_train))
y_pred = ada.predict(X_test)
print(y_pred)

ada.estimator_errors

error = accuracy_score(y_test, y_pred)
print(error)


results = models_evaluation.models_evaluation(args.data, X, y, folds = 10)
print(results)
