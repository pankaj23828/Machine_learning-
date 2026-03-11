import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
y = np.array([0,0,0,0,1,1,1,1])

model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)

print(model.coef_)
print(model.intercept_)
print(y_pred)

plt.scatter(X, y)
plt.plot(X, model.predict_proba(X)[:,1])
plt.show()
