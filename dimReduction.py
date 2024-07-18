# -*- coding: utf-8 -*-
# зменшення розмірності
# див. також кластеризацію
import numpy as np
import matplotlib.pyplot as plt
plt.axis('equal')

from sklearn.decomposition import PCA

# дані з шумом
x1 = np.linspace(10,20,1000)
x2=10+2*x1+1*np.random.normal(size=x1.size)
x = np.vstack([x1, x2]).T # усі ознаки

# аналіз головних компонентів
# шукає напрямки максимальної дисперсії - головні компоненти
model = PCA(n_components=2) # залишити 2 головних компонента
model.fit(x) # підгонка моделі
X = model.transform(x) # перетворити дані
C=model.components_ # напрямки максимальної дисперсії компонентів
a0,b0=C[0] # напрямок (в системі координат x1,x2) максимальної дисперсії головного компонента
a1,b1=C[1] # напрямок (в системі координат x1,x2) максимальної дисперсії другого компонента
V=model.explained_variance_ # відповідні дисперсії
S = V**0.5 # стандартні відхилення
m0,m1=model.mean_ # емпіричні середні

plt.scatter(x1, x2, c=x1) # початкові дані
# напрямки максимальної дисперсії головних компонентів
plt.arrow(m0, m1, S[0]*a0, S[0]*b0, width=.1, head_width=.5, color='k')
plt.arrow(m0, m1, S[1]*a1, S[1]*b1, width=.1, head_width=.5, color='k')
plt.xlabel('x1'),plt.ylabel('x2')
plt.show(); plt.figure()

plt.scatter(X[:,0], X[:,1], c=x1) # перетворені дані
plt.xlabel('X1'),plt.ylabel('X2')
plt.show(); plt.figure()

# аналіз головних компонентів для видалення шуму
model = PCA(n_components=1) # залишити 1 головний компонент
model.fit(x)
X = model.transform(x)
xi=model.inverse_transform(X) # зворотна трансформація в початковий простір ознак x1, x2 (відміна обертання)
plt.scatter(xi[:,0], xi[:,1], c=x1) # дані без шуму
plt.xlabel('x1'),plt.ylabel('x2')
plt.show(); plt.figure()

# факторизація невід'ємних матриць
# дані з шумом
x1 = np.linspace(10,20,1000)
x2=10+2*x1+1*np.random.normal(size=x1.size)
x3=x1+x2+1*np.random.normal(size=x1.size) # сума x1+x2
# x повинні бути невід'ємні !
x = np.vstack([x1, x2, x3]).T # усі ознаки
from sklearn.decomposition import NMF
model = NMF(n_components=2) #
model.fit(x) # підгонка моделі (x - невід'ємні!)
# дозволяє виділити доданки x3
X = model.transform(x)
X = model.inverse_transform(X)
plt.scatter(X[:,0], X[:,1], c=x1)
plt.xlabel('X1'),plt.ylabel('X2')
plt.show(); plt.figure()

# t-SNE намагається знайти двовимірне представлення даних, яке зберігає відстані між точками найкращим чином
# використовують для двовимірного представлення даних
from sklearn.manifold import TSNE
model = TSNE()
X = model.fit_transform(x)
plt.scatter(X[:,0], X[:,1], c=x1) # перетворені дані
plt.xlabel('X1'),plt.ylabel('X2')
plt.show()