# -*- coding: utf-8 -*-
# кодування категоріальних ознак в неперервні
import numpy as np
import pandas as pd
x1 = [0, 2, 2, 3, 9] # неперервні ознаки
x2 = ['Male', 'Female', 'Male', 'Male', 'Male'] # категоріальні ознаки
dataSet = zip(x1,x2) # підготувати дані
df = pd.DataFrame(data = dataSet, columns=['X1', 'X2']) # об'єкт DataFrame
dfc = pd.get_dummies(df) # кодувати категоріальні ознаки
print dfc
X=dfc.values # масив numpy

# кодування неперервних ознак в категоріальні (біннінг)
bins = np.linspace(0, 10, 6)
x1c = np.digitize(x1, bins=bins) # повертає індекси бінів
print x1c
