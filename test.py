print("GreenGuide project started")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Input features
X = data[['N','P','K','temperature','humidity','ph','rainfall']]

# Output label
y = data['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nEnter soil values for crop prediction")

n = float(input("Enter Nitrogen: "))
p = float(input("Enter Phosphorus: "))
k = float(input("Enter Potassium: "))
temp = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
ph = float(input("Enter pH: "))
rain = float(input("Enter Rainfall: "))

prediction = model.predict([[n, p, k, temp, humidity, ph, rain]])

print("Recommended Crop:", prediction[0])