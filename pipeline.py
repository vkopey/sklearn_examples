# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

x = np.array([8,0,3,4,9,7,1,6,3,9])[:, None]
y = np.array([9,0,2,6,9,8,2,9,4,9])

model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(x,y)
print model.predict([[5]])
print model.score(x,y)
step0=model.steps[0][-1]
print step0.get_feature_names()
step1=model.steps[1][-1]
a,b = step1.coef_, step1.intercept_
# перевірка
print a[0]+a[1]*5+a[2]*5**2+b