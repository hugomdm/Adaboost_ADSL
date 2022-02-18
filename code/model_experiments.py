#------ lib packages 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

#----------- external imports 

import Adaboost

#----------- code

np.random.seed(1234)

def changing_parameters(data_type, X, y):
    """
        Function to compare our Adaboost implementation with other Machine Learning models
        Parameters: 
            data_type: binary or multiclass problem
            X: features for training
            y: target feature
        Return: 
            parameters: returns which parameters gave the highest accuracy
            max_accuracy: the value of the highest accuracy
    """
    results, names = list(), list()
    #setting the number of estimators to iterate
    n_estimators = [50,100,200,400,600,800,1000]
    #setting the max_depth between 1 and 2
    for depth in tqdm(range(1,3)):
        for estimator in n_estimators:
            # define base model
            if data_type == 'binary': 
                model = Adaboost.BinaryClassAdaboost(estimator,max_depth= depth)
            if data_type == 'multiclass': 
                model = Adaboost.MultiClassAdaBoost(estimator, max_depth=depth)
            #evaluate the model and collect the results
            scores = cross_validate(model, X, y, cv=10, scoring='accuracy') 
            results.append(np.mean(scores['test_score']))
            names.append("M_D: " + str(depth)+ "\n N_E: " +str(estimator) )

    #plot the Accuracy x Parameters linear graph and save
    x = list(range(0, len(results)))
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(8,7))
    ax.plot(x, results,  linewidth=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=14)
    ax.set_title('Testing different parameters',fontsize= 16) # title of plot
    ax.set_xlabel('Parameters',fontsize = 15) #xlabel
    ax.set_ylabel('Accuracy', fontsize = 15)#ylabel
    ax.tick_params(axis='y', labelsize=14)
    ax.axvline(x=np.argmax(results), color='red', zorder=2)
    fig.tight_layout()
    fig.savefig('../img/parameters_'+str(data_type)+'.png')
    print("Choosing the best parameters - Plot was saved in imgs folder")

    return names[np.argmax(results)], max(results)

def models_comparision(my_ada_model, X, y, folds):
    """
        Function to compare our Adaboost implementation with other Machine Learning models
        Parameters: 
            my_ada_model: our Adaboost initialization
            X: features for training
            y: target feature
            folds: number of folds for cross-validation
        Return: 
            result: dataframe comparing all the solution by performance metrics
    """

    #defining the evaluation metrics
    scores = ['accuracy','precision_macro','recall_macro','f1_macro',]
    # Instantiate the machine learning classifiers
    log_model = LogisticRegression(max_iter=10000)
    svc_model = LinearSVC(dual=False)
    rfc_model = RandomForestClassifier()
    ada_model = AdaBoostClassifier(DecisionTreeClassifier( max_depth=1), n_estimators=600)
    

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

def comparing_adaboost_sklearn(my_ada_model, X, y):
    """
        Function to compare our Adaboost implementation with other Machine Learning models
        Parameters: 
            my_ada_model: our Adaboost initialization
            X: features for training
            y: target feature
        Return: 
            my_classification : Classification report using our Adaboost implementation
            my_confusion: Confusion matrix result using our Adaboost implementation
            sklearn_classification: Classification report using sklearn Adaboost implementation
            sklearn_confusion: Confusion matrix result using sklearn Adaboost implementation
    """

    #doing a 10-fold cross validation with our Adaboost implemetation
    y_pred = cross_val_predict(my_ada_model, X, y, cv=10)
    #classification report and confusion matrix results 
    my_classification = classification_report(y, y_pred)
    my_confusion = confusion_matrix(y, y_pred)

    #getting the estimated errors from each weak learner in the training processs
    #plotting and saving a graph to better vizualize
    my_ada_model.fit(X,y)
    plt.figure(figsize=(8,6))
    plt.plot(list(range(0, len(my_ada_model.estimator_errors))), my_ada_model.estimator_errors)
    plt.title("")
    plt.xlabel('Number of estimators')
    plt.ylabel('ylabel')
    plt.savefig('../img/my_ada_error')
    
    #initiliazing the Scikit Learn Adaboost 
    estimator = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=600)
    #doing a 10-fold cross validation 
    y_pred = cross_val_predict(estimator, X, y, cv=10)
    #classification report and confusion matrix results 
    sklearn_classification = classification_report(y, y_pred)
    sklearn_confusion = confusion_matrix(y, y_pred)

    #getting the estimated errors from each weak learner in the sklearn Adaboost training processs
    #plotting and saving a graph to better vizualize
    estimator.fit(X,y)
    plt.figure(figsize=(8,6))
    plt.plot(list(range(0, len(estimator.estimator_errors_ ))), estimator.estimator_errors_ )
    plt.title("")
    plt.xlabel('Number of estimators')
    plt.ylabel('ylabel')
    plt.savefig('../img/ada_sklearn_error.png')


    return my_classification, my_confusion, sklearn_classification, sklearn_confusion


