# -*- coding: utf-8 -*-
"""
Матриця помилок для бінарної класифікації

|                |спрогнозований клас N  |спрогнозований клас P  |
|----------------|-----------------------|-----------------------|
|фактичний клас N|TN                     |FP                     |
|фактичний клас P|FN                     |TP                     |
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# дві ознаки класів
x=np.array([[0,1,1,2,2,3,2,3,1,3, 6,5,6,7,7,8,7,7,8,5],
            [1,1,3,1,2,2,3,4,4,8, 5,7,6,7,6,7,5,8,8,1]])
# мітки класів (бінарна класифікація)
y=np.array( [0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1] )
x=x.T

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=11)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train) # виконати навчання
Y_test=knn.predict(x_test) # спрогнозовані класи
print y_test # фактичні тестові класи
print Y_test # спрогнозовані тестові класи

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_test) # матриця помилок
# нехай клас 0 - Negative (N), клас 1 - Positive (P) 
TN=cm[0,0] # фактично 0, спрогнозовано 0 (True Negative)
TP=cm[1,1] # фактично 1, спрогнозовано 1 (True Positive)
FN=cm[1,0] # фактично 1, спрогнозовано 0 (False Negative)
FP=cm[0,1] # фактично 0, спрогнозовано 1 (False Positive)

s=(TP+TN)/cm.sum() # правильність - частка правильно класифікованих
# або
s=knn.score(x_test, y_test) # правильність
p=TP/(TP+FP) # точність - частка TP серед усіх спрогнозованих P
r=TP/(TP+FN) # повнота - частка TP серед усіх фактичних P
f1=2*p*r/(p+r)# F1-міра - гармонічне середнє точності і повноти
# або
from sklearn.metrics import f1_score
f1=f1_score(y_test, Y_test) # F1-міра

from sklearn.metrics import classification_report
print classification_report(y_test, Y_test) # повний звіт по класифікації

knn.predict_proba([[4,4]]) # імовірність класу для 1 точки
y_scores=knn.predict_proba(x_test)[:,1] # імовірності класу 1 тестових даних
Y1=y_scores>0.5 # порогове значення імовірності за замовчуванням, порівняйте з Y_test
Y2=y_scores>0.7 # тепер інша к-ть точок будє належати класу 1, порівняйте з Y1
print classification_report(y_test, Y2) # порівняйте з попереднім звітом

# крива точності-повноти
# будується для різних порогових значень імовірності
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
plt.plot(precision, recall)
plt.xlabel(u"Точність"), plt.ylabel(u"Повнота")
plt.show()

# середня точність класифікатора (площа під кривою точності-повноти)
from sklearn.metrics import average_precision_score
print average_precision_score(y_test, y_scores)
# див. також ROC-криві і AUC [Мюллер с.315]