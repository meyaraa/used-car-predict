import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Data
df = pd.read_csv("data/used_cars.csv")

if 'selling_price' in df.columns:
    df = df.rename(columns={'selling_price': 'price'})

# Filter Outlier Ekstrem Target
df = df[(df['price'] > 500) & (df['price'] < df['price'].quantile(0.99))]

if 'year' in df.columns:
    df['age'] = 2026 - df['year']
    df = df.drop(columns=['year'])

# Drop ID / Leakage Columns jika ada
cols_to_drop = [c for c in ['car_id', 'name', 'vin'] if c in df.columns]
df = df.drop(columns=cols_to_drop)

X = df.drop(columns=['price'])
y = df['price']

# 2. Split Data (SEBELUM Preprocessing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simpan Test Set untuk src/evaluate.py
test_data = pd.concat([X_test, y_test], axis=1)
os.makedirs("data", exist_ok=True)
test_data.to_csv("data/test_set.csv", index=False)

# 3. Pipeline Preprocessing
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'str', 'category']).columns.tolist()

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ]
)

# 4. Bandingkan 3 Algoritma dengan 5-Fold Cross Validation
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

best_score = -float('inf')
best_model_name = None
best_pipeline = None

print("=== HASIL 5-FOLD CROSS VALIDATION (R2 Score) ===")
for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2')
    mean_score = scores.mean()
    std_score = scores.std()
    print(f"{name}: Mean R2 = {mean_score:.4f} (+/- {std_score:.4f})")
    
    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name
        best_pipeline = pipe

# Fit Model Terbaik pada seluruh Training Set
best_pipeline.fit(X_train, y_train)

# 5. Simpan Artefak Model & Metadata
os.makedirs("models", exist_ok=True)
joblib.dump(best_pipeline, "models/model.joblib")

metadata = {
    "model_type": best_model_name,
    "cv_r2_score": float(best_score),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}

with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\nModel terbaik ({best_model_name}) berhasil disimpan ke models/model.joblib")