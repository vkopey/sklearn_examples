# -*- coding: utf-8 -*-
"""
# Лінійна регресія
Шукає лінійну залежність у вигляді y=a*x+b методом найменших квадратів.
Див. вибір правильної моделі у scikit-learn:
http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# дані:
x = np.array([8,0,3,4,9,7,1,6,3,9])
y = np.array([9,0,2,6,9,8,2,9,4,9])
x=x[:, None] # або x.reshape(-1,1) або x.reshape(10,1)

model = LinearRegression() # модель лінійна регресія
model.fit(x, y) # підігнати модель (навчання або пошук коефіцієнтів)
print "a=%f b=%f"%(model.coef_[0], model.intercept_) # коефіцієнти моделі
print "R2=%f"%model.score(x,y) # коефіцієнт детермінації

X = np.linspace(0, 10, 100) # нові дані X
Y = model.predict(X[:, None]) # прогноз для X

plt.scatter(x, y) # емпіричні дані
plt.plot(X, Y) # модель
plt.xlabel('x'),plt.ylabel('y')
plt.show()