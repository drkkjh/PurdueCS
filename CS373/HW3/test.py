from cv import CrossValidation
from bagging import Bagging
import numpy as np

cv = CrossValidation(k=4)
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 0, 1])
model = Bagging(2)
model.fit(X, y)
cv.score(X, y, model)