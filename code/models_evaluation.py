
#------ lib packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

#----------- external imports 
import Adaboost

np.random.seed(1234)

# Define the models evaluation function
def models_comparision(data_type:str, X, y, folds):
    
    '''
    X : data set features
    y : data set target
    folds : number of cross-validation folds
    
    '''
    scores = ['accuracy','precision_macro','recall_macro','f1_macro',]
    # Instantiate the machine learning classifiers
    log_model = LogisticRegression(max_iter=10000)
    svc_model = LinearSVC(dual=False)
    rfc_model = RandomForestClassifier()
    ada_model = AdaBoostClassifier()
    if data_type == 'binary': 
        my_ada_model = Adaboost.BinaryClassAdaboost(50)
    if data_type == 'multiclass': 
        my_ada_model = Adaboost.MultiClassAdaBoost(50)

    # Perform cross-validation to each machine learning classifier
    log = cross_validate(log_model, X, y, cv=folds, scoring=scores)
    svc = cross_validate(svc_model, X, y, cv=folds, scoring=scores)
    rfc = cross_validate(rfc_model, X, y, cv=folds, scoring=scores)
    ada = cross_validate(ada_model, X, y, cv=folds, scoring=scores)
    my_ada = cross_validate(my_ada_model, X, y, cv=folds, scoring=scores)

    
    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({'Logistic Regression':[log['test_accuracy'].mean(),
                                                               log['test_precision_macro'].mean(),
                                                               log['test_recall_macro'].mean(),
                                                               log['test_f1_macro'].mean()],
                                       
                                      'Support Vector Classifier':[svc['test_accuracy'].mean(),
                                                                   svc['test_precision_macro'].mean(),
                                                                   svc['test_recall_macro'].mean(),
                                                                   svc['test_f1_macro'].mean()],
                                       
                                      'Random Forest':[rfc['test_accuracy'].mean(),
                                                       rfc['test_precision_macro'].mean(),
                                                       rfc['test_recall_macro'].mean(),
                                                       rfc['test_f1_macro'].mean()],
                                       
                                       'Adaboost Classifier':[ada['test_accuracy'].mean(),
                                                              ada['test_precision_macro'].mean(),
                                                              ada['test_recall_macro'].mean(),
                                                              ada['test_f1_macro'].mean()],
                                        
                                        'My Adaboost Classifier':[my_ada['test_accuracy'].mean(),
                                                              my_ada['test_precision_macro'].mean(),
                                                              my_ada['test_recall_macro'].mean(),
                                                              my_ada['test_f1_macro'].mean()]
                                       },
                                      
                                      index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    
    # Add 'Best Score' column
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
    
    # Return models performance metrics scores data frame
    return models_scores_table

def changing_parameter_max_depth(data_type:str, X, y):

    results = list()
    for i in range(1,11):
		# define base model
        if data_type == 'binary': 
            model = Adaboost.BinaryClassAdaboost(50,max_depth= i)
        if data_type == 'multiclass': 
            model = Adaboost.MultiClassAdaBoost(50, max_depth=i)

        #evaluate the model and collect the results
        scores = cross_validate(model, X, y, cv=10, scoring='accuracy') 
        results.append(np.mean(scores['test_score']))

    plt.figure(figsize=(8,6))
    plt.errorbar(list(range(0, len(results))), results)
    plt.savefig('../img/max_depth_'+str(data_type)+'.png')
    print("Max depth parameter - Plot was saved in imgs folder")
    return np.argmax(results)

def changing_parameter_estimators(data_type:str, X, y):

    results = list()
    n_estimators = [50,100,150,200,400, 500, 700, 1000]
    for i in n_estimators:
		# define base model
        if data_type == 'binary': 
            model = Adaboost.BinaryClassAdaboost(i)
        if data_type == 'multiclass': 
            model = Adaboost.MultiClassAdaBoost(i)

        #evaluate the model and collect the results
        scores = cross_validate(model, X, y, cv=10, scoring='accuracy') 
        results.append(np.mean(scores['test_score']))
    
    plt.figure(figsize=(8,6))
    plt.errorbar(n_estimators, results)
    plt.savefig('../img/estimators_'+str(data_type)+'.png')
    print("Number of estimator parameter - Plot was saved in imgs folder")

    return n_estimators[np.argmax(results)]