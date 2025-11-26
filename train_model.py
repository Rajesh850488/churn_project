# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import os

# Load dataset
DATA_PATH = "data.csv"   # Make sure data.csv is inside your folder

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("data.csv not found! Put it inside your project folder.")

df = pd.read_csv(DATA_PATH)

# Clean target column
if "Churn" not in df.columns:
    raise ValueError("Your dataset must contain a column named 'Churn'")

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop customerID if present
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

X = df.drop(columns=["Churn"])
y = df["Churn"]

# Separate numeric & categorical columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder='passthrough'
)

# Final model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training the model...")
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "churn_model.pkl")
print("Model trained and saved as churn_model.pkl")
