import pandas as pd
import joblib

# Load the trained AI model
model = joblib.load("models/cyber_attack_model.joblib")

# Load one dataset file
df = pd.read_parquet(
    "data/raw/CIC-IDS2017/Benign-Monday-no-metadata.parquet"
)

# Take one network record
sample = df.iloc[[0]].copy()

# Remove the label
X_sample = sample.drop(columns=["Label"])

# Convert data to numbers
X_sample = X_sample.apply(pd.to_numeric, errors="coerce")

# Replace missing values
X_sample = X_sample.fillna(0)

# Make prediction
prediction = model.predict(X_sample)

print("Prediction:", prediction[0])
