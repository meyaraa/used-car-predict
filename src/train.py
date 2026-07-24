import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv("data/used_cars.csv")
df = df.drop_duplicates()

# Preprocessing
df['price'] = df['price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
df['km'] = df['miles'].astype(str).str.replace(' miles', '', regex=False).str.replace(',', '', regex=False).astype(float) * 1.60934
df['age'] = 2026 - df['year']
df['brand'] = df['name'].apply(lambda x: str(x).split(" ")[0])
df['has_accident'] = df['condition'].apply(lambda x: 0 if 'No accidents' in str(x) else 1)
df['is_first_owner'] = df['condition'].apply(lambda x: 1 if '1 Owner' in str(x) else 0)

cols_to_drop = ['name', 'year', 'color', 'condition', 'miles']
df = df.drop(columns=cols_to_drop)

X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Anti-leakage: outlier dibuang HANYA di train set
train_mask = (y_train > 1000) & (y_train < 100000)
X_train = X_train[train_mask]
y_train = y_train[train_mask]

test_data = pd.concat([X_test, y_test], axis=1)
os.makedirs("data", exist_ok=True)
test_data.to_csv("data/test_set.csv", index=False)

numeric_features = ['km', 'age', 'has_accident', 'is_first_owner']
categorical_features = ['brand']

num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer(transformers=[('num', num_transformer, numeric_features), ('cat', cat_transformer, categorical_features)])

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

best_score, best_model_name, best_pipeline = -float('inf'), None, None

print("\n=== MENGHITUNG CROSS-VALIDATION (5-FOLD) ===")
for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2')
    
    # Cetak skor ke layar
    mean_r2 = scores.mean()
    std_r2 = scores.std()
    print(f"{name:17} | Mean R2: {mean_r2:.4f} | Std R2: {std_r2:.4f}")
    
    if mean_r2 > best_score:
        best_score = mean_r2
        best_model_name = name
        best_pipeline = pipe

print("============================================")
print(f"Model terbaik: {best_model_name}\n")

# Latih ulang dengan seluruh data train menggunakan model terbaik
best_pipeline.fit(X_train, y_train)
os.makedirs("models", exist_ok=True)
joblib.dump(best_pipeline, "models/model.joblib")

metadata = {"model_type": best_model_name, "cv_r2_score": float(best_score)}
with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
print("✅ Training selesai dan model tersimpan.")