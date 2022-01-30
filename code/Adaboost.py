# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 15:55:25 2022

@author: Hvins
"""
from sklearn.tree import DecisionTreeClassifier

from math import log, exp


import numpy as np
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
    
    def __init__(self, n_wl:int):
        """
        Initialialisation of Adaboost class
        Parameters: 
            n_wl: int:  count of weaklearner
            type_wl: str: type of weaklearner choosen 
        """
        
        
        self.T = n_wl
        
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
        for t in range(self.T):
            
           
            X_sample, y_sample = self.sampling(X, y, w_t)
            
            #Call Weak learner
            WL = DecisionTreeClassifier(max_depth=1)
            WL.fit(X_sample, y_sample)
            y_pred = WL.predict(X)
            
            #Compute error of weak learner
            eps = self.error_wl(w_t, y_pred, y)
       
            if eps > 0.5:
                break
            
            #Compute weight of weaklearner
            alpha_t = 0.5 * log((1- eps) / eps)
            
            
            #Update weight
            y_temp = np.multiply(y, y_pred)
            y_temp2 = -alpha_t * y_temp 
            w_t = np.multiply(w_t, np.exp(y_temp2))

            #compute zt ????
            z_t =  sum(w_t)
            
            w_t = w_t / z_t
            
            
            beta_t = eps/(1-eps)
            
            w_t = w_t*beta_t
            
            
            self.list_alpha.append(alpha_t)
            self.list_WL.append(WL)
            
            
            
        return 1

    def predict(self, X):
        """
        predict output of Adaboost 
        Paramters: 
            X: array: data
        Return: 
            y_pred: array: data
        """
        
        def sign(x):
            return 1 if x > 0.5 else 0 
        
        def weight(x):
            return np.multiply(x, self.list_alpha)
        
        list_y_pred = []
        
        for WL in self.list_WL:
            list_y_pred.append(WL.predict(X))
            
        arr_y_pred = np.array(list_y_pred)
        
       # arr_y_pred = arr_y_pred * self.list_alpha
        
        arr_y_pred = np.apply_along_axis(weight, 0, arr_y_pred)
    

        y_pred = np.sum(arr_y_pred, axis=0)
        y_pred = np.reshape(y_pred, (y_pred.shape[0],1))
        y_pred = np.apply_along_axis(sign, 1, y_pred )
        
        return y_pred
        
            
        
    def error_wl(self, w_t, y_pred, y):
        """
        error of current weaklearner
        Parameters:
            w_t: array:  weight of observation
            y_pred: array: output of wl 
            y: array: labels
        Return: 
            eps: float: error of wl 
        """
        
        ind_err = []
        for i in range(y_pred.shape[0]):
            if y_pred[i] != y[i]:
                ind_err.append(1) 
            else: 
                ind_err.append(0) 
    
        w_ind_err = np.multiply(w_t,ind_err)
        
        eps = np.sum(w_ind_err)
    
        return eps
    
        
    def sampling(self, X, y, w_t):
        """
        sampling X with w_t 
        Parameters:
            X: array: data
            y: array: labels
            w_t: array: weigth
        Return:
            X_sample: array: sample of X
            y_sample: array: labels corresponding to X_sample
        """
        #put X and y in same array to sample 
        y_temp = np.reshape(y, (y.shape[0], 1))

        data = np.hstack((X, y_temp))
    
        #size of sample
        size = int(0.75*X.shape[0])
        
        #sample
        #sample = choice(data, size, w_t)
        ch = choice([x for x in range(data.shape[0])], size, [1 for x in range(data.shape[0])])
        
        sample = data[ch,:]
        
        y_sample = sample[:,-1]
        X_sample = sample[:,:-1]
        
        return X_sample, y_sample
        
            
            
        
        
      
        
        
        
        
        
        
        
        

        

            
            
            
            
            
    
            
            
            
            
    