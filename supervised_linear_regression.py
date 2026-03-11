
# Supervised Learning Example: Linear Regression
# This script demonstrates how to train and evaluate a linear regression model.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Create Sample Dataset
# -----------------------------
# Example dataset: Hours studied vs Exam score
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Score": [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}

df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[["Hours_Studied"]]
y = df["Score"]

# -----------------------------
# 2. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 4. Make Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 5. Evaluate Model
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Coefficient:", model.coef_[0])
print("Model Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# -----------------------------
# 6. Predict New Value
# -----------------------------
hours = np.array([[7.5]])
predicted_score = model.predict(hours)

print(f"Predicted score for {hours[0][0]} hours of study:", predicted_score[0])
