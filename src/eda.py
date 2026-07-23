import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("reports", exist_ok=True)
df = pd.read_csv("data/used_cars.csv")

if 'selling_price' in df.columns:
    df = df.rename(columns={'selling_price': 'price'})

# 1. Grafik Sebaran Target (Histogram & Log-Transform)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df['price'], kde=True, color='blue')
plt.title("Sebaran Harga Mobil (Original)")

plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df['price']), kde=True, color='green')
plt.title("Sebaran Log(Harga) - Menangani Outlier")
plt.tight_layout()
plt.savefig("reports/target_distribution.png")
plt.close()

# 2. Grafik Data Hilang / Missing Values
plt.figure(figsize=(8, 4))
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    plt.text(0.5, 0.5, 'Tidak Ada Missing Values', ha='center', va='center')
else:
    missing.plot(kind='bar', color='orange')
plt.title("Jumlah Missing Values per Kolom")
plt.ylabel("Jumlah Missing")
plt.tight_layout()
plt.savefig("reports/missing_values.png")
plt.close()

# 3. Fitur Paling Berhubungan (Korelasi & Boxplot Per Kategori)
plt.figure(figsize=(8, 6))
numeric_cols = df.select_dtypes(include=[np.number]).columns
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap Korelasi Fitur Numerik")
plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png")
plt.close()

# 4. Tantangan Wajib: Umur Kendaraan vs Harga (Non-Linearity)
if 'year' in df.columns:
    df['age'] = 2026 - df['year']
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='age', y='price', alpha=0.5)
plt.title("Hubungan Non-Linear: Umur Kendaraan vs Harga")
plt.xlabel("Umur Kendaraan (Tahun)")
plt.ylabel("Harga Jual")
plt.tight_layout()
plt.savefig("reports/age_vs_price_nonlinear.png")
plt.close()

print("Grafik EDA berhasil dibuat di folder reports/")