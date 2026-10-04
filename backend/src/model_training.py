import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/raw/CIC-IDS2017"

# Find all dataset files
files = glob.glob(os.path.join(DATA_PATH, "*.parquet"))

print("Loading datasets...")

dataframes = []

for file in files:
    print("Loading:", os.path.basename(file))
    df = pd.read_parquet(file)
    dataframes.append(df)

# Combine datasets
combined_df = pd.concat(dataframes, ignore_index=True)

print("Dataset shape:", combined_df.shape)

# Create Normal / Attack label
combined_df["Attack_Type"] = combined_df["Label"].apply(
    lambda x: "Normal" if x == "Benign" else "Attack"
)

# Separate input and target
X = combined_df.drop(columns=["Label", "Attack_Type"])
y = combined_df["Attack_Type"]

# Convert any non-numeric columns
X = X.apply(pd.to_numeric, errors="coerce")

# Replace any resulting missing values
X = X.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")

model.fit(X_train, y_train)

print("Training completed!")

# Make predictions
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import joblib

joblib.dump(model, "models/cyber_attack_model.joblib")

print("\nModel saved successfully!")