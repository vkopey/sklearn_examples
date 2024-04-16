# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

X=np.mgrid[0:10:1.0, 0:10:1.0]
x=X.reshape((2, X.size//2))
y = x[0]**2+x[1]**2+10*np.random.normal(size=x[0].shape)
x=x.T
#
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
pipe = make_pipeline(StandardScaler(),PolynomialFeatures(), Ridge())
param_grid = {'polynomialfeatures__degree': [1, 2, 3],
              'ridge__alpha': [0.001, 0.01, 0.1]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=1)
grid.fit(x_train, y_train)

print grid.score(x_test, y_test)
print grid.best_params_
model=grid.best_estimator_

model.fit(x,y) # fit на усіх даних - інша модель!!
s0,s1,s2=model.steps[0][-1],model.steps[1][-1],model.steps[2][-1]
print s1.get_feature_names()
a, b = s2.coef_, s2.intercept_
print a,b

from mpl_toolkits.mplot3d import axes3d
fig = plt.figure() # створити фігуру
ax = fig.add_subplot(111, projection='3d') # добавити графік 3D
Y=model.predict(x) 
Y=Y.reshape(X[0].shape)
ax.scatter(x[:,0], x[:,1], y) # показати емпіричні точки
ax.plot_wireframe(X[0], X[1], Y, rstride=1, cstride=1) # показати теор поверхню
ax.set_xlabel('X0'),ax.set_ylabel('X1'),ax.set_zlabel('Y')
plt.show()

# перевірка в точці [5.,5.]
# Увага! Модель стандартизована
print model.predict([[5.,5.]])
x=s0.transform([[5.,5.]]) # стандартизувати
# або x=([[5.,5.]]-s0.mean_)/s0.scale_
x=x.ravel()
print a[0]+a[1]*x[0]+a[2]*x[1]+a[3]*x[0]**2+a[4]*x[0]*x[1]+a[5]*x[1]**2+b 
