import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Model & Test Set Terpisah (Disentuh HANYA Sekali)
pipeline = joblib.load("models/model.joblib")
test_df = pd.read_csv("data/test_set.csv")

X_test = test_df.drop(columns=['price'])
y_test = test_df['price']

y_pred = pipeline.predict(X_test)

# Hitung Metrik
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("=== EVALUASI MODEL FINAL PADA TEST SET ===")
print(f"MAE  : Rp {mae:,.2f}")
print(f"RMSE : Rp {rmse:,.2f}")
print(f"R2   : {r2:.4f}")

# Plot Error / Residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot - Evaluasi Model Regresi")
plt.tight_layout()
plt.savefig("reports/residuals_plot.png")
plt.close()