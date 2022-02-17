
#------ lib packages 
import sys
import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")


#----------- external imports 
import Adaboost
import data_processing
import model_experiments


arg = sys.argv[1:]
parser = argparse.ArgumentParser(description="Parse command line arguments.")
parser.add_argument("-d", "--data", type=str, required=True,
                    help="Test for binary or multiclass data ['binary', 'multiclass']")

args = parser.parse_args(arg)

X, y = data_processing.split_data(args.data)

print("---------------- Testing different max depth and number of estimator parameters ---------------- ")
print(" ")
params, accuracy = model_experiments.changing_parameters(args.data, X, y)
print("Best parameters value is : " + str(params))
print("Best accuracy value is : " + str(accuracy))

print("---------------- Comparing with different models ---------------- ")

if args.data == 'binary': 
    my_ada_model = Adaboost.BinaryClassAdaboost(600, 1)

if args.data == 'multiclass': 
    my_ada_model = Adaboost.MultiClassAdaBoost(50, 2)
results = model_experiments.models_comparision(my_ada_model, X, y, folds = 10)
print(results)

print(" ")
print("---------------- Comparing our adaboost with Adaboost from Sklearn  ---------------- ")
my_classification, my_confusion, sklearn_classification, sklearn_confusion = model_experiments.comparing_adaboost_sklearn(my_ada_model, X, y)

print("Our results : " + str(my_classification))
print("Our confusion matrix : " + str(my_confusion))
print("Sklearn results : " + str(sklearn_classification))
print("Sklearn confusion matrix : " + str(sklearn_confusion))




