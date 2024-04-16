# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.array([8,0,3,4,9,7,1,6,3,9])
y = np.array([9,0,2,6,9,8,2,9,4,9])
x=x[:, None]

# поліном 2 степні з вільним членом
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
poly.get_feature_names()

model = LinearRegression()
model.fit(x_poly, y)
print model.coef_, model.intercept_, model.score(x_poly,y)

X = np.linspace(0, 10, 1000)
X_poly = poly.transform(X[:, None])
Y = model.predict(X_poly)

plt.scatter(x, y)
plt.plot(X, Y)
plt.ylabel('y'),plt.xlabel('x')
plt.show()