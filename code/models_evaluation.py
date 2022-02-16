
#------ lib packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.ticker as ticker


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

def changing_parameters(data_type:str, X, y):

    results, names = list(), list()
    n_estimators = [50,100, 200,400,600,800,1000]
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

    fig, ax = plt.subplots(figsize=(11,7))
    ax.plot(x, results,  linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45)
    ax.set(title='Testing different parameters', ylabel='Accuracy', xlabel='Parameters')
    #ax.xaxis.set_major_locator(ticker.MaxNLocator(12))
    ax.axvline(x=np.argmax(results),color='red', zorder=2)
    fig.savefig('../img/parameters_'+str(data_type)+'.png')
    print("Choosing the best parameters - Plot was saved in imgs folder")
    return names[np.argmax(results)]
