
#------ lib packages 
import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier

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
    ada = Adaboost.BinaryClassAdaboost(100)

if args.data == 'multiclass': 
    ada = Adaboost.MultiClassAdaBoost(100)
    

ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)

plt.figure(figsize=(8,6))
plt.plot(list(range(0, len(ada.estimator_errors))), ada.estimator_errors)
plt.title("")
plt.xlabel('Number of estimators')
plt.ylabel('ylabel')
plt.savefig('../img/my_ada_error_'+str(args.data)+'.png')

estimator = AdaBoostClassifier(n_estimators=100)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)

plt.figure(figsize=(8,6))
plt.errorbar(list(range(0, len(estimator.estimator_errors_ ))), estimator.estimator_errors_ )
plt.savefig('../img/ada_error_'+str(args.data)+'.png')

error = accuracy_score(y_test, y_pred)
print(error)


results = models_evaluation.models_comparision(args.data, X, y, folds = 10)
print(results)

print("---------------- Testing different max depth and number of estimator parameters ---------------- ")
value = models_evaluation.changing_parameters(args.data, X, y)
print("Best parameters value is : " + str(value))
