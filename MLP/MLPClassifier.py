from sklearn.datasets import load_digits
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier

# =============================================================================
# Classification con red neuronal multicapa
# =============================================================================


X, y = load_digits(return_X_y=True)

#x_plot = np.reshape(X[4,:],(8,8))
#from matplotlib import pyplot as plt
#plt.imshow(x_plot)

sss = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=13)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
# =============================================================================
#  SIEmpre NORMALIZAR EN REDES NEURONALES
# =============================================================================

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)



params = {'hidden_layer_sizes':[(20),(20,20),(50,50),(100,100,100)],
          'activation':['identity','tanh','relu'],
          'alpha':[0.001,0.01]}
# el valor de C se suele poner por decadas
# default kernel is rbf, that is the gaussian one

from sklearn.model_selection import GridSearchCV

miModelo = MLPClassifier()

# =============================================================================
# https://scikit-learn.org/stable/modules/model_evaluation.html
# =============================================================================

gs = GridSearchCV(estimator=miModelo,param_grid=params,scoring='accuracy',cv=5,verbose=1)

gs.fit(X_train,y_train)
resultsCV = gs.cv_results_

clfBest = gs.best_estimator_
clfBest.fit(X_train,y_train)

y_pred = clfBest.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(clfBest,X_test,y_test)

#print the best solution
print(clfBest)

