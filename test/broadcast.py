from sklearn.ensemble import IsolationForest
import numpy as np

x = np.array([1,1.1,1,1,1,1,0.9,1,1,1]).reshape((-1, 1))

clf = IsolationForest()
clf.fit(x)
print(clf.predict(x))