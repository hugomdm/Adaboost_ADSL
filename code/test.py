import Adaboost


from sklearn.datasets import load_breast_cancer



data = load_breast_cancer()


X = data.data

y = data.target

X.shape
y.shape


ada = Adaboost.Adaboost(100)
ada.fit(X, y)
y_pred = ada.predict(X)



2*  a

a = np.array([1,2,3,4])
b = np.array([2,2,2,2])

np.multiply(a,b)
