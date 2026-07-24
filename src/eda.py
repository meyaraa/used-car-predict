import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# folder untuk menyimpan gambar hasil EDA
os.makedirs("reports", exist_ok=True)

print("Memuat dan membersihkan dataset...")
# 1. LOAD & CLEAN DATA
df = pd.read_csv("data/used_cars.csv")
df = df.drop_duplicates()

# Bersihkan kolom harga
df['price'] = df['price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

# Konversi miles ke kilometer (KM) dan buat kolom baru
df['km'] = df['miles'].astype(str).str.replace(' miles', '', regex=False).str.replace(',', '', regex=False).astype(float) * 1.60934

# Ekstraksi fitur lainnya
df['age'] = 2026 - df['year']
df['brand'] = df['name'].apply(lambda x: str(x).split(" ")[0])
df['has_accident'] = df['condition'].apply(lambda x: 0 if 'No accidents' in str(x) else 1)
df['is_first_owner'] = df['condition'].apply(lambda x: 1 if '1 Owner' in str(x) else 0)

# Buang kolom mentah yang sudah tidak dipakai 
cols_to_drop = ['name', 'year', 'color', 'condition', 'miles']
df = df.drop(columns=cols_to_drop)

# Filter Outlier (Harga wajar antara $1.000 - $100.000)
df = df[(df['price'] > 1000) & (df['price'] < 100000)]

# Setel gaya grafik
sns.set_theme(style="whitegrid")

# Grafik 1: Distribusi Harga
print("Membuat grafik 1...")
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True, color='blue')
plt.title('Distribusi Harga Mobil Bekas (USD)', fontsize=14)
plt.xlabel('Harga (USD)')
plt.ylabel('Frekuensi')
plt.savefig('reports/1_distribusi_harga.png', bbox_inches='tight')
plt.close()

# Grafik 2: Hubungan Umur vs Harga
print("Membuat grafik 2...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='price', data=df, alpha=0.5, color='orange')
plt.title('Pengaruh Umur Mobil terhadap Harga', fontsize=14)
plt.xlabel('Umur (Tahun)')
plt.ylabel('Harga (USD)')
plt.savefig('reports/2_umur_vs_harga.png', bbox_inches='tight')
plt.close()

# Grafik 3: Hubungan Jarak Tempuh (KM) vs Harga
print("Membuat grafik 3...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='km', y='price', data=df, alpha=0.5, color='green')
plt.title('Pengaruh Jarak Tempuh (Kilometer) terhadap Harga', fontsize=14)
plt.xlabel('Jarak Tempuh (Kilometer)')
plt.ylabel('Harga (USD)')
plt.savefig('reports/3_km_vs_harga.png', bbox_inches='tight')
plt.close()

# Grafik 4: Dampak Riwayat Tabrakan terhadap Harga
print("Membuat grafik 4...")
plt.figure(figsize=(8, 6))
sns.boxplot(x='has_accident', y='price', data=df, palette='Set2')
plt.title('Perbandingan Harga: Bersih vs Pernah Tabrakan', fontsize=14)
plt.xlabel('Pernah Tabrakan (0 = Tidak, 1 = Ya)')
plt.ylabel('Harga (USD)')
plt.savefig('reports/4_tabrakan_vs_harga.png', bbox_inches='tight')
plt.close()

# Grafik 5: Top 10 Merek Terbanyak
print("Membuat grafik 5...")
plt.figure(figsize=(12, 6))
top_brands = df['brand'].value_counts().nlargest(10)
sns.barplot(x=top_brands.index, y=top_brands.values, palette='viridis')
plt.title('Top 10 Merek Mobil Bekas Terbanyak', fontsize=14)
plt.xlabel('Merek')
plt.ylabel('Jumlah Mobil')
plt.xticks(rotation=45)
plt.savefig('reports/5_top_brands.png', bbox_inches='tight')
plt.close()

# Grafik 6: Heatmap Korelasi Numerik (menggunakan km)
print("Membuat grafik 6...")
plt.figure(figsize=(8, 6))
corr = df[['price', 'km', 'age', 'has_accident', 'is_first_owner']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Heatmap Korelasi Fitur Numerik', fontsize=14)
plt.savefig('reports/6_heatmap_korelasi.png', bbox_inches='tight')
plt.close()

print("✅ SELESAI! Semua grafik telah disimpan di dalam folder 'reports/'")