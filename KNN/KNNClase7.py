# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 18:16:13 2021

@author: franc
"""

# =============================================================================
#                               KNN Clase
# =============================================================================


# =============================================================================
# libraries and data loading
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.model_selection import StratifiedShuffleSplit 
# baraja mi dataset por filas y me da una partición de entrenamiento y una de 
#test un numero d e veces determinado
from sklearn.neighbors import KNeighborsClassifier


X, y = load_iris(return_X_y=True)
Xb, yb = load_boston(return_X_y=True)

# =============================================================================
# Proceso de training estratificación
# =============================================================================


# =============================================================================
# Trainig and prediction
# =============================================================================


sss = StratifiedShuffleSplit(n_splits=10,test_size=0.1,random_state=0)
n_vecinos = 5
accuracies = []

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf = KNeighborsClassifier(n_neighbors=n_vecinos)

    #entreno el algoritmo con las particiones de train
    clf.fit(X_train,y_train)

    # Prediceme ahora la parte de test
    y_pred = clf.predict(X_test)
    
    accuracies.append(100.*sum(y_pred==y_test)/len(y_test))
    
#print numero de aciertos
print("Acierto: ",str(accuracies),"%")
#print media de aciertos
print("Acierto: ",str(np.mean(accuracies)),"%")

# =============================================================================
#  Proceso de training de validación cruzada
# =============================================================================

sss = StratifiedShuffleSplit(n_splits=1,test_size=0.1,random_state=0)
#Se hace para dejar fuera el test
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

params =  {'n_neighbors' : [3,5,9,11,13], 'weights' : ['uniform','distance'],'p' : [1,2,3,4]}

from sklearn.model_selection import GridSearchCV

# EL cros validation te hace un test de todas las convinaciones de param para 
#la que la predicción es mejor

clf = KNeighborsClassifier()

gs = GridSearchCV(estimator=clf,param_grid=params,scoring='accuracy',cv=5)

gs.fit(X_train,y_train)

resultsCV = gs.cv_results_

#cogemos la mejor combinacion
clfBEst = gs.best_estimator_

#entreno con la mejor combinacion
clfBEst.fit(X_train,y_train)

#predigo con este entrenamiento
y_pred = clfBEst.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))


# =============================================================================
# Regresion simple con validación cruzada (boston dataset)
# =============================================================================

# queremos predecir el valor de una casa en una determinada ciudad segén los valores del boston dataset
# Ya no se hace un stratidfied ya que no tenemos categorias asi que vamos a hacer un shuffle split

from sklearn.model_selection import ShuffleSplit

sss = ShuffleSplit(n_splits=1,test_size=0.1,random_state=0)

for train_index, test_index in sss.split(Xb, yb):
    X_train, X_test = Xb[train_index], Xb[test_index]
    y_train, y_test = yb[train_index], yb[test_index]

params =  {'n_neighbors' : [3,5,9,11,13], 'weights' : ['uniform','distance'],'p' : [1,2,3,4]}


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
# EL cros validation te hace un test de todas las convinaciones de param para 
#la que la predicción es mejor

clf = KNeighborsRegressor()

gs = GridSearchCV(estimator=clf,param_grid=params,scoring='neg_mean_absolute_error',cv=5)

gs.fit(X_train,y_train)

resultsCV = gs.cv_results_

#cogemos la mejor combinacion
clfBEst = gs.best_estimator_

#entreno con la mejor combinacion
clfBEst.fit(X_train,y_train)

#predigo con este entrenamiento
y_pred = clfBEst.predict(X_test)

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, y_pred))

plt.scatter(y_test,y_pred,s=60)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)])
plt.show()





