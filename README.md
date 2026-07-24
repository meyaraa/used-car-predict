# Estimasi Harga Kendaraan Bekas 

## Deskripsi Masalah
Proyek ini menyelesaikan masalah untuk sebuah marketplace otomotif. Model regresi dilatih untuk memprediksi harga jual wajar guna menyarankan harga terbaik kepada penjual, dengan mempertimbangkan hubungan non-linear antara umur kendaraan dan harga serta keberadaan outlier harga ekstrem.

## Sumber Data & Lisensi
* **Sumber:** Kaggle — "Used Car Price Prediction" oleh Ayaz (ayaz11)
* **URL:** https://www.kaggle.com/datasets/ayaz11/used-car-price-prediction
* **Nama file asli:** `car_web_scraped_dataset.csv` (di-*rename* menjadi `used_cars.csv` di dalam proyek ini)
* **Lisensi:** Apache 2.0
* **Deskripsi singkat:** Dataset berisi 2.840 baris hasil web-scraping listing mobil bekas, dengan kolom `name, year, miles, color, condition, price` (harga dalam USD).
* **Jumlah baris:** 2.840 (memenuhi syarat minimal 1.000 baris)

## Lingkungan Pengembangan (Versions)
* Python: 3.14.4
* scikit-learn: 1.9.0
* pandas: 3.0.5
* fastapi: 0.110.0

> **Catatan:** Versi di atas harus sama persis dengan yang dipakai saat model dilatih (lihat `models/metadata.json` kunci `sklearn_version`), karena model disimpan sebagai objek pickle yang sensitif terhadap versi library.

## Struktur Data & Model (Tidak Dikomit ke Git)
Folder `data/` dan file `.joblib` di `models/` tidak dikomit ke Git untuk menjaga ukuran repositori tetap ringan dan menghindari penyimpanan data mentah/artefak besar di version control. Penguji dapat memproduksi ulang seluruh artefak secara otomatis dengan menjalankan `src/load_data.py`, `src/train.py`, dan `src/evaluate.py` secara berurutan (lihat langkah di bawah).

## Langkah Menjalankan Proyek (Dari Nol)

1. **Clone repositori & masuk folder**
```bash
   git clone https://github.com/meyaraa/used-car-predict.git
   cd used-car-predict
```

2. **Buat & aktifkan virtual environment**
```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS / Linux
   source .venv/bin/activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
   pip install -r requirements-api.txt
```

4. **Siapkan dataset**
   Buat folder 'data/', Unduh `car_web_scraped_dataset.csv` dari tautan Kaggle di atas, lalu simpan sebagai `data/used_cars.csv`. Lalu jalankan:
```bash
   python src/load_data.py
```
   Skrip ini akan mencetak jumlah baris, kolom, tipe data, dan nilai hilang per kolom sebagai pemeriksaan awal.

5. **Jalankan EDA**
```bash
   python src/eda.py
```
   Menghasilkan 6 grafik di folder `reports/`.

6. **Latih model**
```bash
   python src/train.py
```
   Membandingkan 3 algoritma (Linear Regression, Ridge, Random Forest) dengan 5-fold cross-validation, lalu menyimpan pipeline terbaik ke `models/model.joblib` dan `models/metadata.json`.

7. **Evaluasi model**
```bash
   python src/evaluate.py
```
   Menghitung MAE, RMSE, R² pada test set (disentuh sekali), menganalisis 5 kesalahan prediksi terburuk, dan memperbarui `models/metadata.json`.

8. **Jalankan test otomatis**
```bash
   python -m pytest tests/ -v
```

9. **Jalankan API**
```bash
   uvicorn app.main:app --reload
```
   Buka `http://127.0.0.1:8000/docs` untuk mencoba endpoint secara interaktif.



## Contoh Pemanggilan API

**Request valid (200 OK):**
```bash
curl -X POST "http://127.0.0.1:8000/predict-harga" \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Toyota",
    "age": 4.0,
    "km": 50000.0,
    "has_accident": 0,
    "is_first_owner": 1
  }'
```
Respons:
```json
{
  "status": "success",
  "brand": "Toyota",
  "age": 4.0,
  "estimated_price_usd": 2141.22,
  "estimated_price_idr": 38542000.00,
  "currency_rate_applied": 18000,
  "confidence_margin_pm_idr": 2209000.00,
  "message": "Estimasi harga berhasil dihitung dalam USD dan IDR."
}
```

**Request tidak valid — field hilang (422 Unprocessable Entity):**
```bash
curl -X POST "http://127.0.0.1:8000/predict-harga" \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Toyota",
    "km": 50000.0,
    "has_accident": 0,
    "is_first_owner": 1
  }'
```
Respons:
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "age"],
      "msg": "Field required"
    }
  ]
}
```

## Endpoint Utama
| Endpoint | Method | Deskripsi |
|---|---|---|
| `/` | GET | Informasi layanan |
| `/health` | GET | Status API & status model dimuat |
| `/predict-harga` | POST | Prediksi harga kendaraan bekas |

## Keterbatasan
Dataset tidak memiliki kolom transmisi, jenis bahan bakar, atau kapasitas mesin, sehingga model hanya mengandalkan brand, umur, jarak tempuh, riwayat kecelakaan, dan status kepemilikan pertama sebagai prediktor.
