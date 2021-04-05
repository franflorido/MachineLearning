from sklearn.datasets import load_digits
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC

# =============================================================================
# Classification con SVC
# =============================================================================

X, y = load_digits(return_X_y=True)

#x_plot = np.reshape(X[4,:],(8,8))
#from matplotlib import pyplot as plt
#plt.imshow(x_plot)

sss = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=13)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

params = {'C':[0.1,1,10,100,1000],
          'kernel':['rbf','poly','sigmoid'],
          'degree':[2,3,4],
          'gamma':['scale','auto'],
          'coef0':[0.0]}
# el valor de C se suele poner por decadas
# default kernel is rbf, that is the gaussian one

from sklearn.model_selection import GridSearchCV

miModelo = SVC()

# =============================================================================
# https://scikit-learn.org/stable/modules/model_evaluation.html
# =============================================================================

gs = GridSearchCV(estimator=miModelo,param_grid=params,scoring='accuracy',cv=5,verbose=2)

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

# =============================================================================
# Using PCA to print the dataset
# =============================================================================


from matplotlib import pyplot as plt

plt.figure()

from sklearn.decomposition import PCA

miPCA = PCA(n_components=2)

Xpca = miPCA.fit_transform(X)

plt.scatter(Xpca[:,0],Xpca[:,1],c=y)

# Ver ccon ccuanta información nos hemos quedado
print(miPCA.explained_variance_ratio_)

# =============================================================================
# Now lest work with a SVC model that wotks with this PCA reduced dataset so 
# we can draw the model 
# =============================================================================



for train_index, test_index in sss.split(X, y):
    Xpca_train, Xpca_test = Xpca[train_index], Xpca[test_index]
    y_train, y_test = y[train_index], y[test_index]

params = {'C':[0.1,1,10,100,1000],
          'kernel':['rbf'],
          #'degree':[2,3,4],
          'gamma':['scale','auto'],
          'coef0':[0.0]}
# el valor de C se suele poner por decadas
# default kernel is rbf, that is the gaussian one

from sklearn.model_selection import GridSearchCV

miModelo = SVC()

gs = GridSearchCV(estimator=miModelo,param_grid=params,scoring='accuracy',cv=5,verbose=2)

gs.fit(Xpca_train,y_train)
resultsCV = gs.cv_results_

clfBest = gs.best_estimator_
clfBest.fit(Xpca_train,y_train)

y_pred = clfBest.predict(Xpca_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(clfBest,Xpca_test,y_test)

#print the best solution
print(clfBest)

## COMO SE PUEDE COMPROBAR AL TRABAJAR CON MUCHA MENONS INFORMACIÓN HEMOS PERDIDO BASTANTE ACURACCY

# =============================================================================
#  PLOTTING THE ALGORITHM
# =============================================================================
h=100


# create a mesh to plot in
x_min, x_max = Xpca[:, 0].min() - 1, Xpca[:, 0].max() + 1
y_min, y_max = Xpca[:, 1].min() - 1, Xpca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, h),
                     np.linspace(y_min, y_max, h))


Z = clfBest.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(Xpca[:,0],Xpca[:,1],c=y)





