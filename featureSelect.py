# -*- coding: utf-8 -*-
# відбір ознак
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_regression
#from sklearn.model_selection import train_test_split

x1 = np.linspace(0,10,110) # закономірна ознака
x2 = 5*np.random.normal(size=x1.shape) # шумова ознака
X = np.vstack([x1, x2]).T # усі ознаки
y = 1+2*x1+1*np.random.normal(size=x1.shape)
# на практиці застосовуйте train_test_split

# одновимірний відбір ознак (дисперсійний аналіз)
# за F-значенням вибираємо 50% найбільш значущих ознак
# f_classif - для класифікації
# f_regression - для регресії
select = SelectPercentile(score_func=f_regression, percentile=50)
select.fit(X, y)
print select.scores_ # оцінки ознак
print select.pvalues_ # p-значення оцінок (високі відкидаємо)
print select.get_support() # які ознаки відібрані
# отримуємо новий набір даних без шумових ознак
X_selected = select.transform(X)

# відбір ознак на основі моделі
from sklearn.ensemble import GradientBoostingRegressor
model=GradientBoostingRegressor()
model.fit(X, y)
print model.feature_importances_ # оцінки ознак
# для відбору ознак можна також використовувати коефіцієнти лінійних моделей і моделі Lasso

# або
from sklearn.feature_selection import SelectFromModel
select = SelectFromModel(model) # відбір ознак на основі моделі
select.fit(X, y)
print select.get_support() # які ознаки відібрані

# ітеративний відбір ознак -
# будується послідовність моделей з різною кількістю ознак
from sklearn.feature_selection import RFE
select = RFE(model) # метод рекурсивного виключення ознак
select.fit(X, y)
print select.get_support() # які ознаки відібрані
# застосовуйте також експертні знання для додання нових ознак
