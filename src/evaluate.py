import json
import joblib
import pandas as pd
from datetime import datetime
import sklearn
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

pipeline = joblib.load("models/model.joblib")
test_df = pd.read_csv("data/test_set.csv")

X_test = test_df.drop(columns=['price'])
y_test = test_df['price']
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:,.2f} | RMSE: {rmse:,.2f} | R2: {r2:.4f}")

errors = pd.DataFrame({"Actual": y_test, "Predicted": y_pred, "Abs_Error": abs(y_test - y_pred)})
print("\n=== 5 KESALAHAN PREDIKSI TERBURUK ===")
print(errors.sort_values("Abs_Error", ascending=False).head(5))

with open("models/metadata.json", "r") as f:
    metadata = json.load(f)

metadata.update({
    "test_mae": float(mae),
    "test_rmse": float(rmse),
    "test_r2": float(r2),
    "trained_at": datetime.now().isoformat(),
    "sklearn_version": sklearn.__version__
})

with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
print("✅ Evaluasi selesai dan metadata di-update.")