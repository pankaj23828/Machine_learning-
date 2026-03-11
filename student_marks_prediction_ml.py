import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset
data = {
    'Hours': [1,2,3,4,5,6,7,8,9,10],
    'Marks': [15,20,30,40,50,60,70,80,90,95]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Hours']]
y = df['Marks']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predicted = model.predict([[7]])

print("Predicted Marks for 7 hours study:", predicted)

# Plot graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.show()