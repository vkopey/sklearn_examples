# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neural_network import MLPRegressor

x = np.array([8,0,3,4,9,7,1,6,3,9])[:, None]
y = np.array([9,0,2,6,9,8,2,9,4,9])

# 1 скритий шар
model= MLPRegressor(solver='lbfgs', alpha=0, hidden_layer_sizes=[1], activation='tanh')
model.fit(x, y)
print model.intercepts_# вільні члени
print model.coefs_# коефіцієнти
Y=model.predict(x)

# модель:
h1=np.tanh(-2.73092729+0.75195748*x)
Yf=4.85558639+4.00237182*h1 

# 2 скриті шари
model= MLPRegressor(solver='lbfgs', alpha=0, hidden_layer_sizes=[2], activation='tanh')
model.fit(x, y)
print model.intercepts_
print model.coefs_
Y=model.predict(x)

# модель:
h1=np.tanh(13.87883208-3.47790122*x)
h2=np.tanh(-0.45782002+1.03259976*x)
Yf=-2.89208082*h1+2.111439*h2+3.79645258
