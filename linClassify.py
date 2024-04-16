# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# дві ознаки класів
x=np.array([[0,1,1,2,2,3,2,3,1,3, 6,5,6,7,7,8,7,7,8,5],
            [1,1,3,1,2,2,3,4,4,1, 5,7,6,7,6,7,5,8,8,8]])
# мітки класів (бінарна класифікація)
y=np.array( [0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1] )
x=x.T

# рисуємо матрицю діаграм розсіювання
import pandas as pd
df=pd.DataFrame(x, columns=[0,1])
pd.scatter_matrix(df, c=y, figsize=(5, 5), hist_kwds={'bins': 5})
plt.show()

model=LogisticRegression(C=100) # лінійний классифікатор
model.fit(x,y)
b=model.intercept_ # вільний член
a=model.coef_ # коефіцієнти

# способи прогнозу в точці p
p=np.array([[5,10]]) 
print model.predict(p)
print a[0,0]*p[0,0]+a[0,1]*p[0,1]+b > 0
print model.decision_function(p) > 0

plt.scatter(x[:,0], x[:,1], c=y) # візуалізація класів

# рисуємо границю прийняття рішень
x1, x2 = np.meshgrid(np.linspace(0, 10), np.linspace(0, 10))
xx = np.c_[x1.ravel(), x2.ravel()]
d,l = model.decision_function(xx), [0]
#d,l = model.predict_proba(xx)[:, 1], [0.5] # або
plt.contour(x1, x2, d.reshape(x1.shape) ,levels=l, colors="black")

plt.xlabel('x0'), plt.ylabel('x1')
plt.show()
