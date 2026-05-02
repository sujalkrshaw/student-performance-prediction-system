import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("data/students.csv")

# Features (input)
X = data[["study_hours", "attendance", "assignments", "quiz_score"]]

# Target (output)
y = data["result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(n_estimators=100)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "models/model.pkl")

print("✅ Model saved successfully!")