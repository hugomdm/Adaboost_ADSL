#------ lib packages 
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")


#----------- external imports 
import Adaboost
import data_processing
import model_experiments


"""Main file that starts the work flow by calling all the other functions"""


#--------- Get from command line the arguments for a binary or multiclass problem ------------ #
arg = sys.argv[1:]
parser = argparse.ArgumentParser(description="Parse command line arguments.")
parser.add_argument("-d", "--data", type=str, required=True,
                    help="Test for binary or multiclass data ['binary', 'multiclass']")

args = parser.parse_args(arg)

print("---------------- Pre-processing the data choosen ---------------- ")
print(" ")
if args.data == 'binary': 

    X, y = data_processing.process_data_binary()
    my_ada_model = Adaboost.BinaryClassAdaboost(600, 1)

elif args.data == 'multiclass': 

    X, y = data_processing.process_data_multiclass()
    my_ada_model = Adaboost.MultiClassAdaBoost(50, 2)


print("---------------- Testing different max depth and number of estimator parameters ---------------- ")
print(" ")
params, accuracy = model_experiments.changing_parameters(args.data, X, y)
print(" ")
print("Best parameters value is : " + str(params))
print("Best accuracy value is : " + str(accuracy))

print("---------------- Comparing with different models ---------------- ")
print(" ")
results = model_experiments.models_comparision(my_ada_model, X, y, folds = 10)
print(results)

print("---------------- Comparing our adaboost with Adaboost from Sklearn  ---------------- ")
print(" ")
my_classification, my_confusion, sklearn_classification, sklearn_confusion = model_experiments.comparing_adaboost_sklearn(my_ada_model, X, y)

print("Our results : " + str(my_classification))
print("Our confusion matrix : " + str(my_confusion))
print(" ")
print("Sklearn results : " + str(sklearn_classification))
print("Sklearn confusion matrix : " + str(sklearn_confusion))




