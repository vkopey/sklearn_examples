# -*- coding: utf-8 -*-
"""
Після оптимізації параметрів моделі потрібно її перевірити на екзаменаційних даних. Для цього на початку дані треба ділити на дві частини. Перша буде використовуватись для оптимізації, а друга - для перевірки найкращої моделі.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

x = np.array([8,0,3,4,9,7,1,6,3,9])[:, None]
y = np.array([9,0,2,6,9,8,2,9,4,9])
# розділити дані випадково
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

pipe = make_pipeline(StandardScaler(),PolynomialFeatures(), Ridge())
param_grid = {'polynomialfeatures__degree': [1, 2, 3],
              'ridge__alpha': [0.001, 0.01, 0.1]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=1)
grid.fit(x_train, y_train)#Увага!!! Усі можливі комбінації параметрів
print grid.score(x_test, y_test) # або cross_val_score
#grid.fit(x, y) # або використовувати усі дані
print grid.best_params_
