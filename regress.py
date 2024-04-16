# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

x = np.array([8,0,3,4,9,7,1,6,3,9])[:, None]
y = np.array([9,0,2,6,9,8,2,9,4,9])
# на практиці застосовуйте train_test_split

M=[] # моделі
M+=[LinearRegression()]
M+=[KNeighborsRegressor()]
M+=[SVR()]
M+=[DecisionTreeRegressor()]
M+=[GradientBoostingRegressor()]
M+=[MLPRegressor()]

for model in M:
    model.fit(x, y)
    print model.predict([[5]]), model.score(x,y)