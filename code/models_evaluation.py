
#------ lib packages 

import pandas as pd
from sklearn.model_selection import cross_validate

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
#----------- external imports 
import Adaboost

# Define the models evaluation function
def models_evaluation(data_type:str, X, y, folds):
    
    '''
    X : data set features
    y : data set target
    folds : number of cross-validation folds
    
    '''
    scores = ['accuracy','precision_macro','recall_macro','f1_macro',]
    # Instantiate the machine learning classifiers
    log_model = LogisticRegression(max_iter=10000)
    svc_model = LinearSVC(dual=False)
    dtr_model = DecisionTreeClassifier()
    rfc_model = RandomForestClassifier()
    gnb_model = GaussianNB()
    ada_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4))
    if data_type == 'binary': 
        my_ada_model = Adaboost.BinaryClassAdaboost(100)
    if data_type == 'multiclass': 
        my_ada_model = Adaboost.MultiClassAdaboost(100)

    # Perform cross-validation to each machine learning classifier
    log = cross_validate(log_model, X, y, cv=folds, scoring=scores)
    svc = cross_validate(svc_model, X, y, cv=folds, scoring=scores)
    dtr = cross_validate(dtr_model, X, y, cv=folds, scoring=scores)
    rfc = cross_validate(rfc_model, X, y, cv=folds, scoring=scores)
    gnb = cross_validate(gnb_model, X, y, cv=folds, scoring=scores)
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
                                       
                                      'Decision Tree':[dtr['test_accuracy'].mean(),
                                                       dtr['test_precision_macro'].mean(),
                                                       dtr['test_recall_macro'].mean(),
                                                       dtr['test_f1_macro'].mean()],
                                       
                                      'Random Forest':[rfc['test_accuracy'].mean(),
                                                       rfc['test_precision_macro'].mean(),
                                                       rfc['test_recall_macro'].mean(),
                                                       rfc['test_f1_macro'].mean()],
                                       
                                      'Gaussian Naive Bayes':[gnb['test_accuracy'].mean(),
                                                              gnb['test_precision_macro'].mean(),
                                                              gnb['test_recall_macro'].mean(),
                                                              gnb['test_f1_macro'].mean()], 
                                       
                                       
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
    return(models_scores_table)


    #sns.countplot(x = heart_data['HeartDisease'], data = heart_data, palette='rocket')
