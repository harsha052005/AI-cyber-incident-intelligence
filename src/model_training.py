import pandas as pd
import glob
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/raw/CIC-IDS2017"

files = glob.glob(os.path.join(DATA_PATH, "*.parquet"))

print("Loading datasets...")

dataframes = []

for file in files:
    print("Loading:", os.path.basename(file))
    df = pd.read_parquet(file)
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)

print("Dataset shape:", combined_df.shape)

combined_df["Attack_Type"] = combined_df["Label"].apply(
    lambda x: "Normal" if x == "Benign" else "Attack"
)

X = combined_df.drop(columns=["Label", "Attack_Type"])
y = combined_df["Attack_Type"]

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

print("Creating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")

model.fit(X_train, y_train)

print("Training completed!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import joblib

joblib.dump(model, 'models/cyber_attack_model.joblib')

print('\nModel saved successfully!')
