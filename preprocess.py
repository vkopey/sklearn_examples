# -*- coding: utf-8 -*-
# підготовка даних і маштабування
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

x = np.array([8,0,3,4,9,7,1,6,3,9])[:, None]
y = np.array([9,0,2,6,9,8,2,9,4,9])

S=[]
S+=[MinMaxScaler()] # масштабує в діапазоні 0..1
S+=[StandardScaler()] # середнє 0, дисперсія 1
S+=[RobustScaler()] # те саме що StandardScaler, але ігнорує викиди
for scaler in S:
    scaler.fit(x) # отримати модель для маштабування
    x_scaled=scaler.transform(x) # масштабувати
    print x_scaled
    print scaler.inverse_transform(x_scaled) # зворотне перетворення

# Увага! Завжди застосовуйте fit для навчаючих даних
# і потім transform для навчаючих і тестових даних

# Більшість моделей працює краще, якщо ознаки і залежна змінна мають нормальний розподіл.
# Часто (особливо під час обробки дискретних даних) функції log і exp дозволяють досягти більш симетричного розподілу [Мюллер с.249, 98]. Застосовуйте їх для лінійних моделей, але не для моделей на основі дерев.