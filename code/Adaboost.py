# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 15:55:25 2022

@author: Hvins
"""

import random 

from sklearn.tree import DecisionTreeClassifier

from math import log, e

from numpy.random import choice



# Algo Adaboost: 

# Fit :     
# Pour chaque arbre : 
    
#     Calculer y_pred avec un weak learner
    
#     Calculer l'erreur global du wl 
    
#     Calculer son poids dans le vote final 
    
#     Calculer les nouveaux poids des X 
    
#     Normaliser les poids 
    
    
# Predict:
    
#     Calculer y_pred pour chaque arbre 
    
#     pondérer la décision 

            
                            
    

class Adaboost():
    """
    """
    
    def __init__(self, n_wl:int, type_wl:str):
        """
        Initialialisation of Adaboost class
        Parameters: 
            n_wl: int:  count of weaklearner
            type_wl: str: type of weaklearner choosen 
        """
        
        
        self.T = n_wl
        
        self.WL = Weaklearer(type_wl)
        
        
        self.list_WL = [] #list with model
        self.list_alpha = [] #list with weight of model 
        
        
        
    def fit(self, X, y):
        """
        Fit model 
        Parameters: 
            X: array: data
            y: array: vector of class labels where yi E Y= {1,..., k}
        """
        self.list_WL = [] 
        self.list_alpha = []
        ##Initialize weight: 
        m = y.shape[0]                
        w = []
        w_t = [1/m for x in range(X.shape[0])]        
        
        #Construction des weaklearner
        for t in self.T:
            
           
            X_sample, y_sample = sampling(X, y, wt)
            
            #Call Weak learner
            WL = DecisionTreeClassifier()
            WL.fit(X_sample, y_sample, max_depth=1)
            y_pred = WL.predict(X)
            
            #Compute error of weak learner
            error = error_wl(w_t, y_pred, y)
       
            if error > 0.5:
                break
            
            #Compute weight of weaklearner
            alpha_t = 0.5 * log((1- error) / error)
            
            
            #Update weight
            w_t = w_t * e(-alpha_t * y * y_pred)

            #compute zt ????
            z_t =  sum(w_t)
            
            w_t = w_t / z_t
            
            
            betat = eps/(1-eps)
            
            wt = wt*betat
            
            
            self.list_alpha.append(alpha_t)
            self.list_WL.append(WL)
            

    def predict(X):
        """
        predict output of Adaboost 
        Paramters: 
            X: array: data
        Return: 
            y_pred: array: data
        """
        
        def sign(x):
            return 1 if x > 0.5 else 0 
        
        list_y_pred = []
        
        for WL in self.list_WL:
            list_y_pred.append(WL.predict(X))
            
        arr_y_pred = np.array(list_y_pred)
        
        arr_y_pred = arr_y_pred * self.list_alpha

        y_pred = np.sum(arr_y_pred, axis=0)
        
        y_pred = npp.apply_along_axis(sign 1, y_pred )
        
        return y_pred
        
            
        
    def error_wl(wt, y_pred, y):
        """
        error of current weaklearner
        Parameters:
            wt: array:  weight of observation
            y_pred: array: output of wl 
            y: array: labels
        Return: 
            err_wl: float: error of wl 
        """
        
        ind_err = []
        for i in y_pred.shape[0]:
            if y_pred[i] != y[i]
                ind_err.append(1) 
            else: 
                ind_err.append(0) 
    
        w_ind_err = wt*ind_err
        
        error = sum(w_ind_err)
    
        return error
    
        
    def sampling(X, y, wt):
        """
        sampling X with wt 
        Parameters:
            X: array: data
            y: array: labels
            wt: array: weigth
        Return:
            X_sample: array: sample of X
            y_sample: array: labels corresponding to X_sample
        """
        #put X and y in same array to sample 
        data = np.append(X, y, axis=1)
        
        #size of sample
        size = int(0.75*X.shape[0])
        
        #sample
        sample = choice(data, size, wt)
        
        y_sample = sample[:,-1]
        X_sample = sample[:,:-1]
        
        return X_sample, y_sample
        
            
            
        
        
      
        
        
        
        
        
        
        
        

        

            
            
            
            
            
    
            
            
            
            
    