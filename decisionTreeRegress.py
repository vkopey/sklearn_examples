# -*- coding: utf-8 -*-
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_graphviz

x = np.array([8,0,3,4,9,7,1,6,3,9])[:, None]
y = np.array([9,0,2,6,9,8,2,9,4,9])

model= DecisionTreeRegressor(max_depth=2)
model.fit(x, y)
print model.predict([[5]]), model.score(x,y)
print model.feature_importances_

from StringIO import StringIO
f = StringIO()
export_graphviz(model, out_file=f)
print f.getvalue()
path=model.decision_path([[5]]).toarray()
