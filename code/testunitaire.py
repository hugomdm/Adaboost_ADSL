import Adaboost

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

data = load_breast_cancer()


X = data.data

y = data.target

X.shape
y.shape


ada = Adaboost.BinaryClassAdaboost(50)
ada.fit(X, y)
y_pred = ada.predict(X)

error = ada.metrics_adaboost(y, y_pred)



accuracy_score(y, y_pred)