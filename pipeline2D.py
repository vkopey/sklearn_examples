# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

X=np.mgrid[0:10:1.0, 0:10:1.0]
x=X.reshape((2, X.size//2))
y = x[0]**2+x[1]**2+10*np.random.normal(size=x[0].shape)
x=x.T
model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(x, y)

s0,s1=model.steps[0][-1],model.steps[1][-1]
print s0.get_feature_names()
a,b=s1.coef_, s1.intercept_
# перевірка
print model.predict([[5.,5.]])
print a,b
print a[0]+a[1]*5+a[2]*5+a[3]*25+a[4]*25+a[5]*25+b 
