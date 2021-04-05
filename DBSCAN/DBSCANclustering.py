# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 19:03:44 2021

@author: franc
"""
# =============================================================================
#                                   DBSCAN
# =============================================================================

# =============================================================================
# Libraries
# =============================================================================


from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


# =============================================================================
# A partir de ahora solo voy a jugar con la x para hacer un clustering para
# intentar adivinar la y
# =============================================================================

X, y = load_iris(return_X_y=True)

# X es la matriz de entrenamiento e y es la salida que tiene que tener 

miModelo = DBSCAN(eps = 0.5, min_samples = 3)
miModelo.fit(X)
clusters = miModelo.labels_

# imprimimos los dos para ver si se parece
y
clusters

# =============================================================================
# Hacemos plots para ver los resultados
# =============================================================================

#plot del resultado en y que nos da la el dataset
plt.figure()
plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],c=y)
plt.title('Color = y')

#plot del resultado obtenido con el clusterin de KMeans
plt.subplot(1,2,2)
plt.scatter(X[:,0],X[:,1],c=clusters)
plt.title('Color = clusters')

# =============================================================================
# Vamos a pintar reduciendo la dimensionalidad con PCA
# =============================================================================

miPCA = PCA(n_components = 2)

Xtr = miPCA.fit_transform(X)

#plot del resultado en y que nos da la el dataset
plt.figure()
plt.subplot(1,2,1)
plt.scatter(Xtr[:,0],Xtr[:,1],c=y)
plt.title('Color = y')

#plot del resultado obtenido con el clusterin de KMeans
plt.subplot(1,2,2)
plt.scatter(Xtr[:,0],Xtr[:,1],c=clusters)
plt.title('Color = clusters')

#Vamos a comprobar las medidas que me dice que un cluster es bueno En este caso la medida de la silueta

from sklearn.metrics import silhouette_score

sc = silhouette_score(X,clusters) # me da un valor de 0.39, donde el mejor valor es 1 y el peor -1





