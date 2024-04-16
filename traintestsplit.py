# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x = np.array([8,0,3,4,9,7,1,6,3,9])[:, None]
y = np.array([9,0,2,6,9,8,2,9,4,9])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)

model = LinearRegression()
model.fit(x_train, y_train)
print model.coef_, model.intercept_
print model.score(x_train, y_train)

y_pred = model.predict(x_test)
print model.score(x_test, y_test) # точність моделі на тестових даних
print r2_score(y_test, y_pred) # або
print model.score(x, y) 
#plt.scatter(expected, predicted)