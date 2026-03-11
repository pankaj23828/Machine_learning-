
# House Price Prediction (Simple ML Project)

import pandas as pd
from sklearn.linear_model import LinearRegression

# ----------------------
# Dataset
# ----------------------
data = {
    "area": [1000, 1500, 1800, 1200, 2000, 2200],
    "bedrooms": [2, 3, 4, 2, 4, 5],
    "age": [5, 10, 8, 3, 12, 6],
    "price": [3000000, 4500000, 5000000, 3500000, 5500000, 6000000]
}

df = pd.DataFrame(data)

# ----------------------
# Features and Target
# ----------------------
X = df[["area", "bedrooms", "age"]]
y = df["price"]

# ----------------------
# Train Model
# ----------------------
model = LinearRegression()
model.fit(X, y)

print("Model Trained Successfully")

# ----------------------
# User Input
# ----------------------
area = int(input("Enter house area (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
age = int(input("Enter house age: "))

# ----------------------
# Prediction
# ----------------------
prediction = model.predict([[area, bedrooms, age]])

print("Predicted House Price:", int(prediction[0]))
