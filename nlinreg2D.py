# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.array([[0,1,2,0,1,2,0,1,2],
              [0,0,0,1,1,1,2,2,2]])
y = np.array([0,1,2,1,2,3,2,3,9])
x=x.T

# поліном 2 степні з вільним членом
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
poly.get_feature_names()

model = LinearRegression()
model.fit(x_poly, y)
print model.coef_, model.intercept_, model.score(x_poly,y)

X=np.mgrid[0:3:0.5,0:3:0.5]
X_=X.reshape((2,X.size//2))
X_poly = poly.transform(X_.T)
Y = model.predict(X_poly)

from mpl_toolkits.mplot3d import axes3d
fig = plt.figure() # створити фігуру
ax = fig.add_subplot(111, projection='3d') # добавити графік 3D
Y=Y.reshape(X.shape[1:])
ax.scatter(x[:,0], x[:,1], y) # показати емпіричні точки
ax.plot_wireframe(X[0], X[1], Y, rstride=1, cstride=1) # показати теор поверхню
ax.set_xlabel('X0'),ax.set_ylabel('X1'),ax.set_zlabel('Y')
plt.show()
