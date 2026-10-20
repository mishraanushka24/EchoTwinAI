
import pandas as pd

sessions = [
    {
        "clicks": 5,
        "errors": 0,
        "time_seconds": 60,
        "backtracks": 0,
        "help_requests": 0,
        "task_completed": 1,
        "friction": 0
    },
    {
        "clicks": 12,
        "errors": 3,
        "time_seconds": 180,
        "backtracks": 4,
        "help_requests": 2,
        "task_completed": 0,
        "friction": 1
    },
    {
        "clicks": 7,
        "errors": 1,
        "time_seconds": 90,
        "backtracks": 1,
        "help_requests": 0,
        "task_completed": 1,
        "friction": 0
    },
    {
        "clicks": 15,
        "errors": 4,
        "time_seconds": 240,
        "backtracks": 5,
        "help_requests": 3,
        "task_completed": 0,
        "friction": 1
    },
    {
        "clicks": 9,
        "errors": 2,
        "time_seconds": 150,
        "backtracks": 3,
        "help_requests": 1,
        "task_completed": 1,
        "friction": 0
    },
    {
        "clicks": 18,
        "errors": 6,
        "time_seconds": 300,
        "backtracks": 7,
        "help_requests": 4,
        "task_completed": 0,
        "friction": 1
    }
]

df = pd.DataFrame(sessions)

print("EchoTwin AI - User Session Dataset")
print(df)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Input features
X = df[
    [
        "clicks",
        "errors",
        "time_seconds",
        "backtracks",
        "help_requests",
        "task_completed"
    ]
]

# Target label
y = df["friction"]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)

# Create the machine learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

print("\nModel Training Completed!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

import joblib

# Save the trained model
joblib.dump(model, "friction_model.pkl")

print("\nModel saved successfully!")
print("File name: friction_model.pkl")

