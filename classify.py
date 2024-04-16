# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# дві ознаки класів
x=np.array([[0,1,1,2,2,3,2,3,1,3, 6,5,6,7,7,8,7,7,8,5],
            [1,1,3,1,2,2,3,4,4,8, 5,7,6,7,6,7,5,8,8,1]])
# мітки класів (бінарна класифікація)
y=np.array( [0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1] )
x=x.T

plt.scatter(x[:,0], x[:,1], c=y) # візуалізація класів
plt.xlabel('x0'), plt.ylabel('x1')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=11)
print y_test # фактичні тестові класи

M=[] # моделі
M+=[KNeighborsClassifier(n_neighbors=3, weights='distance')]
# метод k сусідів
# n_neighbors - к-ть сусідів
# weights - функція ваг
M+=[LogisticRegression(C=100, penalty="l1")]
# логістична регресія
# C - параметр регуляризації (менше C - більша регуляризація). За замовчуванням 1
# penalty - тип регуляризації
M+=[LinearSVC(C=100)]
# лінійний метод опорних векторів
M+=[GaussianNB()]
# наївний баєсів класифікатор
# див. також MultinomialNB і BernoulliNB
M+=[DecisionTreeClassifier(max_depth=4)]
# дерево рішень
# max_depth - максимальна глибина дерева
M+=[RandomForestClassifier(n_estimators=5)]
# випадковий ліс
# n_estimators - кількість дерев
# див. також GradientBoostingClassifier
M+=[SVC(kernel='rbf', C=10, gamma=0.1)]
# ядерний метод опорних векторів
M+=[MLPClassifier(solver='lbfgs', hidden_layer_sizes=[2], activation='tanh',alpha=0.1)]
# багатошаровий перцептрон
# hidden_layer_sizes=[2] - кількість елементів в скритому шарі
# activation - функція активації
# alpha - регуляризація
# застосовуйте StandardScaler

for model in M:
    model.fit(x_train, y_train) # виконати навчання
    print model.predict(x_test), model.score(x_test, y_test) # спрогнозовані класи
    