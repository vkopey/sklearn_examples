# -*- coding: utf-8 -*-
"""
# Лінійні моделі методом регуляризації
З метою зменшення перенавчання намагаються зменшити коефіцієнти моделі (кут нахилу лінії)
шляхом збільшення параметра регуляризації alpha>=0.
alpha = 0 відповідає звичайній LinearRegression.
https://ru.wikipedia.org/wiki/%D0%A0%D0%B5%D0%B3%D1%83%D0%BB%D1%8F%D1%80%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F_(%D0%BC%D0%B0%D1%82%D0%B5%D0%BC%D0%B0%D1%82%D0%B8%D0%BA%D0%B0)
Для пошуку найкращого значення alpha можуть бути застосовані RidgeCV або ElasticNetCV.
"""
import numpy as np
from sklearn.linear_model import Ridge, Lasso, RidgeCV, ElasticNet

# дані
x = np.array([8,0,3,4,9,7,1,6,3,9])[:, None]
y = np.array([9,0,2,6,9,8,2,9,4,9])

# лінійна регресія методом регуляризації Тихонова
model = Ridge(alpha = 0.5)
model.fit(x, y)
print model.coef_, model.intercept_, model.score(x,y)
##
# інший метод регуляризації - деякі коефіцієнти моделі можуть бути рівні 0
model = Lasso(alpha = 0.5)
model.fit(x, y)
print model.coef_, model.intercept_, model.score(x,y)
##
# комбінована L1 і L2 регуляризація
# якщо l1_ratio = 1 то це L1 регуляризація
model = ElasticNet(alpha=0.5, l1_ratio = 0.5)
model.fit(x, y)
print model.coef_, model.intercept_, model.score(x,y)
##
# лінійна регресія Ridge (автоматично знаходить найкраще alpha)
model = RidgeCV(alphas=[0.1, 1.0, 10.0])
model.fit(x, y)
print model.coef_, model.intercept_, model.score(x,y)
print model.alpha_
# див. також ElasticNetCV