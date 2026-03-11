import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print(model.coef_)
print(model.intercept_)

plt.scatter(X, y)
plt.plot(X, y_pred)
plt.show()
