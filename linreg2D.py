# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([[0,1,2,0,1,2,0,1,2],
              [0,0,0,1,1,1,2,2,2]])
y = np.array([0,1,2,1,2,3,2,3,9])
x=x.T
model = LinearRegression()
model.fit(x, y)

X=np.mgrid[0:3:0.5,0:3:0.5]
X_=X.reshape((2,X.size//2))
Y = model.predict(X_.T)
print model.coef_, model.intercept_, model.score(x,y)

from mpl_toolkits.mplot3d import axes3d
fig = plt.figure() # створити фігуру
ax = fig.add_subplot(111, projection='3d') # додати графік 3D
Y=Y.reshape(X.shape[1:])
ax.scatter(x[:,0], x[:,1], y) # показати емпіричні точки
ax.plot_wireframe(X[0], X[1], Y, rstride=1, cstride=1) # показати теоретичну поверхню
ax.set_xlabel('X0'),ax.set_ylabel('X1'),ax.set_zlabel('Y')
plt.show()