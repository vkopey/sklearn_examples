# -*- coding: utf-8 -*-
"""
Після оптимізації параметрів моделі потрібно її перевірити на екзаменаційних даних. Для цього на початку дані треба ділити на дві частини. Перша буде використовуватись для оптимізації, а друга - для перевірки найкращої моделі.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# емпіричні дані
x = np.array([0,1,2,3,4,5,6,7,8,9,10])[:, None]
y = np.array([0,1,2,3,4,5,6,7,8,9,30])

# пошук найкращого параметра alpha
alphas = np.logspace(-1, 2, 100) # масив параметрів регуляризації
scores = []
for alpha in alphas: # для кожного alpha
    # середня правильність моделі шляхом перехресної перевірки
    s=cross_val_score(Ridge(alpha), x, y, cv=3).mean()
    scores.append([s,alpha]) # додати у список
maxscore,alpha=max(scores, key=lambda s:s[0])
print maxscore,alpha # найбільша правильність і відповідне alpha

import matplotlib.pyplot as plt
plt.plot(alphas, [s[0] for s in scores]) # залежність score від alpha
plt.xlabel('alpha'),plt.ylabel('score')
plt.show()

# або
from sklearn.model_selection import GridSearchCV
model = GridSearchCV(Ridge(), dict(alpha=alphas), cv=3)
# виконує перехрестну перевірку (CV) для кожного елементу alphas 
model.fit(x, y)
print model.best_score_ # середня правильність CV моделі, яка побудована на навчальних даних CV
print model.best_params_
model.best_estimator_ # найкраща модель
print model.score(x,y) # правильність найкращої моделі
model.cv_results_ # усі результати пошуку

# або
from sklearn.linear_model import RidgeCV
model = RidgeCV(alphas=alphas, cv=3)
model.fit(x, y)
print model.alpha_

# або рандомізований пошук найкращих параметрів
from sklearn.model_selection import RandomizedSearchCV
model = RandomizedSearchCV(Ridge(), dict(alpha=alphas), cv=3, n_iter=10)
model.fit(x, y)
print model.best_score_
print model.best_params_

# вкладена перехресна перевірка [Мюллер с.292]
scores = cross_val_score(RidgeCV(alphas=alphas, cv=3), x, y, cv=3)
print scores.mean()
