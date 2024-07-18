# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# випадкові дані
# x з 0 до 10
x = 10*np.random.random((30, 1))
# y = a*x**2 + b з шумом
y = 0.2*x**2+1 + 2*np.random.normal(size=x.shape)
plt.scatter(x, y)
plt.xlabel('x'), plt.ylabel('y')
plt.show(); plt.figure()

# будуємо криві перевірки
model = make_pipeline(PolynomialFeatures(), LinearRegression())
degree = np.arange(0, 8)
train_scores, test_scores = validation_curve(model, x, y, 'polynomialfeatures__degree', degree, cv=3)
plt.plot(degree, np.mean(train_scores, 1), 'o-') # оцінка навчання
plt.plot(degree, np.mean(test_scores, 1), 'o--') # оцінка перевірки
plt.xlabel('degree'),plt.ylabel('score')
plt.show(); plt.figure()

# будуємо криві навчання для моделі degree=2
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
train_sizes, train_scores, test_scores = learning_curve(model, x, y, cv=3, train_sizes=np.linspace(0.2, 1, 20))
plt.plot(train_sizes, np.mean(train_scores, 1), 'o-')# оцінка навчання
plt.plot(train_sizes, np.mean(test_scores, 1), 'o--')# оцінка перевірки
plt.xlabel('train_size'),plt.ylabel('score')
plt.show()