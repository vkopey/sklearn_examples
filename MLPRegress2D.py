# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neural_network import MLPRegressor

x = np.array([[0,1,2,0,1,2,0,1,2],
              [0,0,0,1,1,1,2,2,2]])
y = np.array([0,1,2,1,2,3,2,3,9])
x=x.T

# 1 скритий шар розміром 1
model= MLPRegressor(solver='lbfgs', alpha=0, hidden_layer_sizes=[1], activation='tanh')
model.fit(x, y)
print model.intercepts_ # вільні члени
print model.coefs_ # коефіцієнти
Y=model.predict(x)

# модель:
h1=np.tanh(4.6427557-0.54709987*x[:,0]-0.54759059*x[:,1])
Yf=560.91609523-560.27879759*h1

