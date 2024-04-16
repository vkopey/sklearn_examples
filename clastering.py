# -*- coding: utf-8 -*-
# кластеризація
import numpy as np
import matplotlib.pyplot as plt
plt.axis('equal')
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

# дані
# дві ознаки класів
x=np.array([[0,1,1,2,2,3,2,3,1,3, 6,5,6,7,7,8,7,7,8,5],
            [1,1,3,1,2,2,3,4,4,8, 5,7,6,7,6,7,5,8,8,1]])
# мітки класів
y=np.array( [0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1] )
x=x.T

plt.scatter(x[:,0], x[:,1], c=y) # візуалізація класів
plt.xlabel('x0'), plt.ylabel('x1')
plt.show()

m=KMeans(n_clusters=2) # метод k-середніх
# алгоритм обчислює центри ваги кластерів
m.fit(x)
print m.labels_
print m.predict([[1,2]]) # прогноз в новій точці
print m.cluster_centers_ # центри кластерів

m=AgglomerativeClustering(n_clusters=2, linkage='ward') # агломеративна кластеризація
# обєднує подібні кластери в агломерації
# linkage - критерій порівняння кластерів
# агломеративні методи не мають методу predict
m.fit(x)
print m.labels_
# див. також ієрархічну класифікацію [Мюллер, с. 201]
#from scipy.cluster.hierarchy import dendrogram, ward

m=DBSCAN(eps=2.0, min_samples=5) # оснований на щільності алгоритм кластеризації просторових даних з наявністю шуму
# шукає ядрові точки в щільних зонах
# точка є ядровою, якщо min_samples точок знаходяться в її околі радіусом eps
# ядрові точки в околі eps утворюють кластер
m.fit(x)
print m.labels_ # шумові точки позначаються (-1)

#оцінка якості кластеризації
from sklearn.metrics.cluster import adjusted_rand_score
print adjusted_rand_score(y, m.labels_)