
#------ lib packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#----------- external imports 
import Adaboost

np.random.seed(1234)

# Define the models evaluation function
def models_comparision(my_ada_model, X, y, folds):
    
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

def changing_parameters(data_type, X, y):

    results, names = list(), list()
    n_estimators = [50,100,200,400,600,800,1000]
    #n_estimators = [50,200]
    for i in tqdm(range(1,3)):
        for j in n_estimators:
            # define base model
            if data_type == 'binary': 
                model = Adaboost.BinaryClassAdaboost(j,max_depth= i)
            if data_type == 'multiclass': 
                model = Adaboost.MultiClassAdaBoost(j, max_depth=i)
            #evaluate the model and collect the results
            scores = cross_validate(model, X, y, cv=10, scoring='accuracy') 
            results.append(np.mean(scores['test_score']))
            names.append("M_D: " + str(i)+ "\n N_E: " +str(j) )

    x = list(range(0, len(results)))
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(11,7), frameon=False)
    ax.plot(x, results,  linewidth=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45)
    ax.set(title='Testing different parameters', ylabel='Accuracy', xlabel='Parameters')
    ax.axvline(x=np.argmax(results), color='red', zorder=2)
    fig.savefig('../img/parameters_'+str(data_type)+'.png')
    print("Choosing the best parameters - Plot was saved in imgs folder")
    return names[np.argmax(results)], max(results)


def comparing_adaboost_sklearn(my_ada_model, X, y):



    y_pred = cross_val_predict(my_ada_model, X, y, cv=10)
    my_classification = classification_report(y, y_pred)
    my_confusion = confusion_matrix(y, y_pred)

    my_ada_model.fit(X,y)
    plt.figure(figsize=(8,6))
    plt.plot(list(range(0, len(my_ada_model.estimator_errors))), my_ada_model.estimator_errors)
    plt.title("")
    plt.xlabel('Number of estimators')
    plt.ylabel('ylabel')
    plt.savefig('../img/my_ada_error')
    
    estimator = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=600)
    y_pred = cross_val_predict(estimator, X, y, cv=10)

    estimator.fit(X,y)
    plt.figure(figsize=(8,6))
    plt.plot(list(range(0, len(estimator.estimator_errors_ ))), estimator.estimator_errors_ )
    plt.title("")
    plt.xlabel('Number of estimators')
    plt.ylabel('ylabel')
    plt.savefig('../img/ada_sklearn_error.png')

    sklearn_classification = classification_report(y, y_pred)
    sklearn_confusion = confusion_matrix(y, y_pred)


    return my_classification, my_confusion, sklearn_classification, sklearn_confusion


