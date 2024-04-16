# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

x = np.array([8,0,3,4,9,7,1,6,3,9])[:, None]
y = np.array([9,0,2,6,9,8,2,9,4,9])
model = LinearRegression()

# перехресна перевірка - це удосконалення train_test_split + score
s=cross_val_score(model, x, y, cv=3)
print s, s.mean()

# перехресна перевірка з випадковими перестановками
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(test_size=0.5, n_splits=3) # спробуйте test_size=3
s=cross_val_score(model, x, y, cv=cv)
print s, s.mean()
for train_index, test_index in cv.split(x):
    print train_index, test_index # вивести індекси даних для CV