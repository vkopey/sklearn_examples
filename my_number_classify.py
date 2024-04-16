# -*- coding: utf-8 -*-
import numpy as np
zero="""
.xxx
.x.x
.x.x
.xxx

xxxx
x..x
x..x
xxxx

xxx.
x.x.
x.x.
xxx.

xxx.
x.x.
xxx.
....

.xxx
.x.x
.xxx
....

....
xxx.
x.x.
xxx.

....
.xxx
.x.x
.xxx

.xx.
x..x
x..x
.xx.

..x.
.x.x
.x.x
..x.

.x..
x.x.
x.x.
.x..
"""

one="""
...x
...x
...x
...x

..x.
..x.
..x.
..x.

.x..
.x..
.x..
.x..

x...
x...
x...
x...

..xx
...x
...x
...x

...x
..xx
...x
...x

.xx.
..x.
..x.
..x.

xx..
.x..
.x..
xxx.

.xx.
..x.
..x.
.xxx

.x..
xx..
.x..
.x..
"""
n=zero+one
n=n.split('\n\n')
x=[]
for i in n:
    s=i.replace('\n','').replace('.','0').replace('x','1')
    x.append(np.array([int(e) for e in s]))
x=np.array(x)
y=np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
model = KNeighborsClassifier()
model.fit(X_train, y_train)
print model.score(X_test, y_test)
print model.predict([[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1]])
