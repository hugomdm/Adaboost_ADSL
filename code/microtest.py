# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:00:58 2022

@author: Hvins
"""

import Adaboost

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np



#binary 
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.7)

ada = Adaboost.BinaryClassAdaboost(50)
ada.fit(X_train, y_train)

y_pred = ada.predict(X_test)
accuracy_score(y_test, y_pred)


#multicalss 
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.7)

ada = Adaboost.MultiClassAdaBoost(50)
ada.fit(X_train, y_train)

y_pred, y_pred_score = ada.predict(X_test)
accuracy_score(y_test, y_pred)

a = np.zeros((2,3))
b = [1,2]

for e, bi in enumerate(b):
    a[e,bi] += 1
    print(b)
    
print(a)

1.0 // 1
