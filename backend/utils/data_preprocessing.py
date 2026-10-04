import pandas as pd
import glob
import os

DATA_PATH = "data/raw/CIC-IDS2017"

files = glob.glob(os.path.join(DATA_PATH, "*.parquet"))

print("Dataset files found:", len(files))

dataframes = []

for file in files:
    print("Loading:", os.path.basename(file))
    df = pd.read_parquet(file)
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)

print("Combined dataset shape:", combined_df.shape)
print("\nX columns:")
print(X.columns.tolist())
X = combined_df.drop(columns=['Label', 'Attack_Type'])
y = combined_df['Attack_Type']

print('\nX shape:', X.shape)
print('y shape:', y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)